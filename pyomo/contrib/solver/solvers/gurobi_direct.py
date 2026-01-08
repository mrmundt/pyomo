#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import datetime
import io
import math
import operator
import os
import logging
import time
from typing import Optional, Tuple

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.common.enums import ObjectiveSense
from pyomo.common.errors import MouseTrap
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

from pyomo.contrib.solver.common.base import SolverBase, Availability, _LicenseManager
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase

logger = logging.getLogger(__name__)

gurobipy, gurobipy_available = attempt_import('gurobipy')

# Historically, we stored a live gurobipy.Env object on the class as a shared
# singleton (per-process) for all solver instances. That made dill/pickle of
# the class fail, because gurobipy.Env is a Cython type with a non-trivial
# __cinit__ and no pickling support.
#
# To keep classes picklable (e.g., for multiprocessing),
# we store only a small, picklable key on the class, and put the
# real Env object in this module-global registry.
_GUROBI_ENV_REGISTRY = {}          # { "GLOBAL": gurobipy.Env }
_GUROBI_ENV_KEY = "GLOBAL"         # single shared env per process
_GUROBI_GLOBAL_CLIENTS = 0         # total clients across all solver instances


class GurobiConfigMixin:
    """
    Mixin class for Gurobi-specific configurations
    """

    def __init__(self):
        self.use_mipstart: bool = self.declare(
            'use_mipstart',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Gurobi.",
            ),
        )


class GurobiConfig(BranchAndBoundConfig, GurobiConfigMixin):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        BranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        GurobiConfigMixin.__init__(self)


class GurobiDirectSolutionLoader(SolutionLoaderBase):
    def __init__(self, grb_model, grb_cons, grb_vars, pyo_cons, pyo_vars, pyo_obj):
        self._grb_model = grb_model
        self._grb_cons = grb_cons
        self._grb_vars = grb_vars
        self._pyo_cons = pyo_cons
        self._pyo_vars = pyo_vars
        self._pyo_obj = pyo_obj
        GurobiDirect._register_env_client()

    def __del__(self):
        if python_is_shutting_down():
            return
        # Free the associated model
        if self._grb_model is not None:
            self._grb_cons = None
            self._grb_vars = None
            self._pyo_cons = None
            self._pyo_vars = None
            self._pyo_obj = None
            # explicitly release the model
            self._grb_model.dispose()
            self._grb_model = None
        # Release the gurobi license if this is the last reference to
        # the environment (either through a results object or solver
        # interface)
        GurobiDirect._release_env_client()

    def load_vars(self, vars_to_load=None, solution_number=0):
        assert solution_number == 0
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, map(operator.attrgetter('x'), self._grb_vars))
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        for p_var, g_var in iterator:
            p_var.set_value(g_var, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        assert solution_number == 0
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, map(operator.attrgetter('x'), self._grb_vars))
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        return ComponentMap(iterator)

    def get_duals(self, cons_to_load=None):
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoDualsError()

        def dedup(_iter):
            last = None
            for con_info_dual in _iter:
                if not con_info_dual[1] and con_info_dual[0] is last:
                    continue
                last = con_info_dual[0]
                yield con_info_dual

        iterator = dedup(
            zip(self._pyo_cons, map(operator.attrgetter('Pi'), self._grb_cons))
        )
        if cons_to_load:
            cons_to_load = set(cons_to_load)
            iterator = filter(
                lambda con_info_dual: con_info_dual[0] in cons_to_load, iterator
            )
        return {con_info: dual for con_info, dual in iterator}

    def get_reduced_costs(self, vars_to_load=None):
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoReducedCostsError()

        iterator = zip(self._pyo_vars, map(operator.attrgetter('Rc'), self._grb_vars))
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_rc: var_rc[0] in vars_to_load, iterator)
        return ComponentMap(iterator)


class _GurobiLicenseManager(_LicenseManager):
    """
    License handler for Gurobi instances. Handles checkout, locking,
    and release.

    The manager:
        - creates the Env on first use (or reuses a live one)
        - increments a per-class client count
        - and decrements it on exit, closing the Env when the last client leaves

    The actual Env object lives in _GUROBI_ENV_REGISTRY, keyed by class
    """

    def __init__(self):
        super().__init__()
        self._local_client_count = 0  # per-instance reentrancy depth

    def acquire(
        self,
        timeout: Optional[float] = None,
        retry_timeout: Optional[float] = float("inf"),
    ) -> None:
        """
        Acquire (or reuse) the shared Gurobi Env.

        timeout:        Max time to wait for a *single* Env() construction attempt.
                        If None/0, try once (no per-attempt timeout)
        retry_timeout:  Total wall time to keep retrying if license temporarily
                        unavailable. Default: keep trying (block) until it succeeds.
        """
        # Re-entrant acquire for this solver instance: no-op except for depth
        if self._local_client_count > 0:
            self._local_client_count += 1
            return

        # Try reusing an existing live env
        env = self._ensure_env()
        if env is not None:
            self._register_global_client()
            self._local_client_count = 1
            return

        # No live env yet: create with retry/backoff
        start = time.time()
        sleep_for = 0.1
        while True:
            try:
                self._set_env(gurobipy.Env())
                self._register_global_client()
                self._local_client_count = 1
                return
            except Exception as e:
                # If we aren't allowed to retry, or exceeded retry budget: give up
                now = time.time()
                if retry_timeout is not None and (now - start) >= retry_timeout:
                    logger.warning(
                        "Gurobi license acquire failed after %.2fs: %s",
                        now - start, e, exc_info=True
                    )
                    raise
                logger.info(
                    "Gurobi license not acquired yet; retrying: %s", e, exc_info=True
                )
                # Respect single-attempt timeout by sleeping in small chunks
                # (we don't have a direct way to pass timeout into Env(); this
                # loop is a coarse-grained backoff).
                time.sleep(min(sleep_for, 2.0))
                sleep_for = min(sleep_for * 2, 2.0)

    def release(self) -> None:
        """Release one local (per-instance) hold; close shared Env when last global client leaves."""
        if self._local_client_count <= 0:
            # Be forgiving (user called release without acquire)
            logger.debug("GurobiLicenseManager.release() called with no outstanding acquire.")
            return

        self._local_client_count -= 1
        if self._local_client_count > 0:
            return  # still held by this instance

        # Transition from 1 -> 0 for this solver instance: drop a GLOBAL ref
        self._release_global_client()

    # Expose a safe getter for the shared env (for availability checks, etc.)
    @staticmethod
    def _get_env():
        env = _GUROBI_ENV_REGISTRY.get(_GUROBI_ENV_KEY)
        if env is None:
            return None
        # If the env looks dead (no _cenv), treat as closed
        if not hasattr(env, "_cenv"):
            try:
                env.close()
            except Exception:
                pass
            _GurobiLicenseManager._set_env(None)
            return None
        return env

    @staticmethod
    def _set_env(env):
        if env is None:
            _GUROBI_ENV_REGISTRY.pop(_GUROBI_ENV_KEY, None)
        else:
            _GUROBI_ENV_REGISTRY[_GUROBI_ENV_KEY] = env

    @classmethod
    def _ensure_env(cls):
        env = cls._get_env()
        if env is not None:
            return env
        return None

    @classmethod
    def _register_global_client(cls):
        global _GUROBI_GLOBAL_CLIENTS
        _GUROBI_GLOBAL_CLIENTS += 1

    @classmethod
    def _release_global_client(cls):
        global _GUROBI_GLOBAL_CLIENTS
        if _GUROBI_GLOBAL_CLIENTS > 0:
            _GUROBI_GLOBAL_CLIENTS -= 1

        env = cls._get_env()
        if _GUROBI_GLOBAL_CLIENTS <= 0 and env is not None:
            if _GUROBI_GLOBAL_CLIENTS < 0:
                logger.warning(
                    "Gurobi global env client refcount went negative (%s). "
                    "This should be reported to the Pyomo dev team.",
                    _GUROBI_GLOBAL_CLIENTS,
                )
            try:
                env.close()
            except Exception as err:
                logger.warning("Exception while closing Gurobi environment: %r", err)
            finally:
                cls._set_env(None)


class GurobiSolverMixin:
    """
    gurobi_direct and gurobi_persistent check availability and set versions
    in the same way. This moves the logic to a central location to reduce
    duplicate code.
    """
    _available_cache = None
    _version_cache = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # Replace the default _LicenseManager with the Gurobi-specific one
        self.license = _GurobiLicenseManager()

    def _is_gp_available(self) -> bool:
        try:
            # this triggers the deferred import
            return bool(gurobipy_available)
        except Exception:
            return False

    def available(self, recheck: bool = False) -> Availability:
        """
        Best-effort classification:
          - NotFound         : gurobipy not importable
          - FullLicense      : check succeeds on a full-size model (>2000 vars)
          - LimitedLicense   : check triggers limit (e.g., demo/community)
          - LicenseError     : denial/timeout/bad/unknown licensing states
        """
        if not recheck and self._available_cache is not None:
            return self._available_cache
        # Note that we set the _available_cache on the *most derived
        # class* and not on the instance, or on the base class.  That
        # allows different derived interfaces to have different
        # availability (e.g., persistent has a minimum version
        # requirement that the direct interface doesn't)
        if not self._is_gp_available():
            self.__class__._available_cache = Availability.NotFound
        else:
            with capture_output(capture_fd=True):
                try:
                    with self.license():
                        status = self._check_license_status()
                except Exception as e:
                    logger.debug(
                        "License check failed in available(): %s", e, exc_info=True
                    )
                    status = Availability.LicenseError

            self.__class__._available_cache = status
        return self._available_cache

    def version(self, recheck: bool = False) -> Optional[Tuple[int, int, int]]:
        if not recheck and self._version_cache is not None:
            return self._version_cache

        if not self._is_gp_available():
            return None

        self.__class__._version_cache = (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )
        return self._version_cache

    def _check_license_status(self) -> Availability:
        """
        Build a tiny model (>2000 vars) to test demo/community limits and
        classify license level.
        """
        env = None
        m = None
        try:
            env = gurobipy.Env()
            env.setParam("OutputFlag", 0)
            m = gurobipy.Model(env=env)
            m.Params.OutputFlag = 0
            m.addVars(range(2001))
            m.optimize()
            return Availability.FullLicense
        except gurobipy.GurobiError as e:
            msg = (str(e) or "").lower()
            errno = getattr(e, "errno", None)
            if errno in (10010,) or "too large" in msg:
                return Availability.LimitedLicense
            if (
                "no gurobi license" in msg
                or "not licensed" in msg
                or "license not found" in msg
                or "expired" in msg
                or "queue" in msg
                or "timeout" in msg
                or errno in (10009,)
            ):
                return Availability.LicenseError
            # Treat any other unexpected status as an error
            return Availability.LicenseError
        finally:
            try:
                if m is not None:
                    m.dispose()
            except Exception:
                pass
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass


class GurobiDirect(GurobiSolverMixin, SolverBase):
    """
    Interface to Gurobi using gurobipy
    """

    CONFIG = GurobiConfig()

    _tc_map = None

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()

        timer.start('compile_model')
        repn = LinearStandardFormCompiler().write(
            model, mixed_form=True, set_sense=None
        )
        timer.stop('compile_model')

        if len(repn.objectives) > 1:
            raise IncompatibleModelError(
                f"The {self.__class__.__name__} solver only supports models "
                f"with zero or one objectives (received {len(repn.objectives)})."
            )

        timer.start('prepare_matrices')
        inf = float('inf')
        ninf = -inf
        bounds = list(map(operator.attrgetter('bounds'), repn.columns))
        lb = [ninf if _b is None else _b for _b in map(operator.itemgetter(0), bounds)]
        ub = [inf if _b is None else _b for _b in map(operator.itemgetter(1), bounds)]
        CON = gurobipy.GRB.CONTINUOUS
        BIN = gurobipy.GRB.BINARY
        INT = gurobipy.GRB.INTEGER
        vtype = [
            (
                CON
                if v.is_continuous()
                else BIN if v.is_binary() else INT if v.is_integer() else '?'
            )
            for v in repn.columns
        ]
        sense_type = list('=<>')
        sense = [sense_type[r[1]] for r in repn.rows]
        timer.stop('prepare_matrices')

        ostreams = [io.StringIO()] + config.tee
        res = Results()

        orig_cwd = os.getcwd()
        try:
            if config.working_dir:
                os.chdir(config.working_dir)

            # Acquire a Gurobi env for the duration of solve (opt + postsolve):
            with self.license():
                env = self.license._get_env()

                with capture_output(TeeStream(*ostreams), capture_fd=False):
                    gurobi_model = gurobipy.Model(env=env)

                    timer.start('transfer_model')
                    x = gurobi_model.addMVar(
                        len(repn.columns),
                        lb=lb,
                        ub=ub,
                        obj=repn.c.todense()[0] if repn.c.shape[0] else 0,
                        vtype=vtype,
                    )
                    A = gurobi_model.addMConstr(repn.A, x, sense, repn.rhs)
                    if repn.c.shape[0]:
                        gurobi_model.setAttr('ObjCon', repn.c_offset[0])
                        gurobi_model.setAttr(
                            'ModelSense', int(repn.objectives[0].sense)
                        )
                    timer.stop('transfer_model')

                    options = config.solver_options
                    gurobi_model.setParam('LogToConsole', 1)

                    if config.threads is not None:
                        gurobi_model.setParam('Threads', config.threads)
                    if config.time_limit is not None:
                        gurobi_model.setParam('TimeLimit', config.time_limit)
                    if config.rel_gap is not None:
                        gurobi_model.setParam('MIPGap', config.rel_gap)
                    if config.abs_gap is not None:
                        gurobi_model.setParam('MIPGapAbs', config.abs_gap)

                    if config.use_mipstart:
                        raise MouseTrap("MIPSTART not yet supported")

                    for key, option in options.items():
                        gurobi_model.setParam(key, option)

                    timer.start('optimize')
                    gurobi_model.optimize()
                    timer.stop('optimize')

                # Build Results while the env is still alive
                res = self._postsolve(
                    timer,
                    config,
                    GurobiDirectSolutionLoader(
                        gurobi_model, A, x, repn.rows, repn.columns, repn.objectives
                    ),
                )

        finally:
            os.chdir(orig_cwd)

        res = self._postsolve(
            timer,
            config,
            GurobiDirectSolutionLoader(
                gurobi_model,
                A,
                x.tolist(),
                list(map(operator.itemgetter(0), repn.rows)),
                repn.columns,
                repn.objectives,
            ),
        )

        res.solver_config = config
        res.solver_name = 'Gurobi'
        res.solver_version = self.version()
        res.solver_log = ostreams[0].getvalue()

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        return res

    def _postsolve(self, timer: HierarchicalTimer, config, loader):
        grb_model = loader._grb_model
        status = grb_model.Status

        results = Results()
        results.solution_loader = loader
        results.timing_info.gurobi_time = grb_model.Runtime

        if grb_model.SolCount > 0:
            if status == gurobipy.GRB.OPTIMAL:
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        results.termination_condition = self._get_tc_map().get(
            status, TerminationCondition.unknown
        )

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if loader._pyo_obj:
            try:
                if math.isfinite(grb_model.ObjVal):
                    results.incumbent_objective = grb_model.ObjVal
                else:
                    results.incumbent_objective = None
            except (gurobipy.GurobiError, AttributeError):
                results.incumbent_objective = None
            try:
                results.objective_bound = grb_model.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                if grb_model.ModelSense == ObjectiveSense.minimize:
                    results.objective_bound = -math.inf
                else:
                    results.objective_bound = math.inf
        else:
            results.incumbent_objective = None
            results.objective_bound = None

        results.extra_info.IterCount = grb_model.getAttr('IterCount')
        results.extra_info.BarIterCount = grb_model.getAttr('BarIterCount')
        results.extra_info.NodeCount = grb_model.getAttr('NodeCount')

        timer.start('load solution')
        if config.load_solutions:
            if grb_model.SolCount > 0:
                results.solution_loader.load_vars()
            else:
                raise NoFeasibleSolutionError()
        timer.stop('load solution')

        return results

    def _get_tc_map(self):
        if GurobiDirect._tc_map is None:
            grb = gurobipy.GRB
            tc = TerminationCondition
            GurobiDirect._tc_map = {
                grb.LOADED: tc.unknown,  # problem is loaded, but no solution
                grb.OPTIMAL: tc.convergenceCriteriaSatisfied,
                grb.INFEASIBLE: tc.provenInfeasible,
                grb.INF_OR_UNBD: tc.infeasibleOrUnbounded,
                grb.UNBOUNDED: tc.unbounded,
                grb.CUTOFF: tc.objectiveLimit,
                grb.ITERATION_LIMIT: tc.iterationLimit,
                grb.NODE_LIMIT: tc.iterationLimit,
                grb.TIME_LIMIT: tc.maxTimeLimit,
                grb.SOLUTION_LIMIT: tc.unknown,
                grb.INTERRUPTED: tc.interrupted,
                grb.NUMERIC: tc.unknown,
                grb.SUBOPTIMAL: tc.unknown,
                grb.USER_OBJ_LIMIT: tc.objectiveLimit,
            }
        return GurobiDirect._tc_map

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import datetime
import io
import math
import os
import logging
import time
from typing import Mapping, Optional, Sequence

from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.common.enums import ObjectiveSense
from pyomo.common.errors import ApplicationError, InfeasibleConstraintException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base import VarData, ConstraintData

from pyomo.contrib.solver.common.base import (
    SolverBase,
    Availability,
    _LicenseManagerBase,
)
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
    get_infeasible_results,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoader

logger = logging.getLogger(__name__)

gurobipy, gurobipy_available = attempt_import('gurobipy')


class GurobiConfig(BranchAndBoundConfig):
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
        self.warmstart_discrete_vars: bool = self.declare(
            'warmstart_discrete_vars',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Gurobi.",
            ),
        )


class GurobiDirectSolutionLoaderBase(SolutionLoader):
    def __init__(self, solver_model, pyomo_model) -> None:
        super().__init__()
        self._solver_model = solver_model
        self._pyomo_model = pyomo_model  # needed for suffixes
        GurobiDirectBase._register_env_client()

    def _get_active_solution_id(self) -> int:
        return self._solver_model.getParamInfo('SolutionNumber')[2]

    def _set_solution_id(self, solution_id: int) -> int:
        previous_id = self._get_active_solution_id()
        self._solver_model.setParam('SolutionNumber', solution_id)
        return previous_id

    def get_number_of_solutions(self) -> int:
        return self._solver_model.SolCount

    def get_solution_ids(self) -> list:
        return list(range(self.get_number_of_solutions()))

    def _get_var_lists(self):
        """
        Should return a list of pyomo vars and a list of gurobipy vars
        """
        raise NotImplementedError('should be implemented by derived classes')

    def _get_var_map(self):
        raise NotImplementedError('should be implemented by derived classes')

    def _get_con_map(self):
        raise NotImplementedError('should be implemented by derived classes')

    def __del__(self):
        # Release the gurobi license if this is the last reference to
        # the environment (either through a results object or solver
        # interface)
        GurobiDirectBase._release_env_client()

    def _get_primals(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> tuple[list[VarData], list[float]]:
        if self._solver_model.SolCount == 0:
            raise NoSolutionError()
        if vars_to_load is None:
            pvars, gvars = self._get_var_lists()
        else:
            pvars = vars_to_load
            gvars = list(map(self._get_var_map().__getitem__, vars_to_load))
        if (
            self._get_active_solution_id()
            and not self._solver_model.getAttr('NumIntVars')
            and not self._solver_model.getAttr('NumBinVars')
        ):
            raise ValueError(
                'Cannot obtain suboptimal solutions for a continuous model'
            )
        if self._get_active_solution_id():
            grbFcn = 'Xn' if gurobipy.GRB.VERSION_MAJOR < 13 else 'PoolNX'
        else:
            grbFcn = 'X'
        vals = self._solver_model.getAttr(grbFcn, gvars)
        return pvars, vals

    def load_vars(self, vars_to_load: Sequence[VarData] | None = None) -> None:
        pvars, vals = self._get_primals(vars_to_load=vars_to_load)
        for pv, val in zip(pvars, vals):
            pv.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        pvars, vals = self._get_primals(vars_to_load=vars_to_load)
        res = ComponentMap(zip(pvars, vals))
        return res

    def _get_rc_all_vars(self):
        pvars, gvars = self._get_var_lists()
        vals = self._solver_model.getAttr("Rc", gvars)
        return ComponentMap((i, val) for i, val in zip(pvars, vals))

    def _get_rc_subset_vars(self, vars_to_load):
        var_map = self._get_var_map()
        gvars = [var_map[i] for i in vars_to_load]
        vals = self._solver_model.getAttr("Rc", gvars)
        return ComponentMap(zip(vars_to_load, vals))

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        if self._get_active_solution_id():
            raise NoReducedCostsError('Can only get reduced costs for solution_id = 0')
        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoReducedCostsError()
        if self._solver_model.IsMIP:
            # this will also return True for continuous, nonconvex models
            raise NoReducedCostsError('Can only get reduced costs for convex problems')
        if vars_to_load is None:
            res = self._get_rc_all_vars()
        else:
            res = self._get_rc_subset_vars(vars_to_load=vars_to_load)
        return res

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> dict[ConstraintData, float]:
        if self._get_active_solution_id():
            raise NoDualsError('Can only get duals for solution_id = 0')
        if self._solver_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoDualsError()
        if self._solver_model.IsMIP:
            # this will also return True for continuous, nonconvex models
            raise NoDualsError('Can only get duals for convex problems')

        qcons = set(self._solver_model.getQConstrs())
        con_map = self._get_con_map()
        if cons_to_load is None:
            cons_to_load = con_map.keys()

        duals = {}
        for c in cons_to_load:
            gurobi_con = con_map[c]
            if type(gurobi_con) is tuple:
                # only linear range constraints are supported
                gc1, gc2 = gurobi_con
                d1 = gc1.Pi
                d2 = gc2.Pi
                if abs(d1) > abs(d2):
                    duals[c] = d1
                else:
                    duals[c] = d2
            else:
                if gurobi_con in qcons:
                    duals[c] = gurobi_con.QCPi
                else:
                    duals[c] = gurobi_con.Pi

        return duals


class _GurobiLicenseManager(_LicenseManagerBase):
    """License manager for the Gurobi solver interfaces.

    In Gurobi, the "license" is controlled by a :class:`gurobipy.Env`, and every
    :class:`gurobipy.Model` is owned by the ``Env`` that created it (closing the
    ``Env`` disposes all of its models). All of the Gurobi interfaces share a
    single, class-level ``Env`` on :class:`GurobiDirectBase`, whose lifetime is
    handled by a reference count of "environment clients". Acquiring a license
    registers one client (creating the shared ``Env``), and releasing it
    unregisters that client (closing the shared ``Env`` once the last client
    goes away).
    """

    #: Number of seconds to wait between retry attempts
    _retry_buffer = 0.1

    def __init__(self, solver) -> None:
        super().__init__()
        self._solver = solver

    def _acquire(
        self,
        timeout: Optional[float] = float("inf"),
        retry_timeout: Optional[float] = None,
    ) -> None:
        # ``timeout`` limits how long we wait for the license server to respond
        # on a single checkout attempt; ``retry_timeout`` limits how long we
        # keep retrying while the checkout keeps failing (e.g., all licenses in
        # use)
        if retry_timeout is not None and retry_timeout != float("inf"):
            deadline = time.monotonic() + retry_timeout
        while True:
            try:
                self._create_env(timeout=timeout)
                GurobiDirectBase._register_env_client()
                return
            except gurobipy.GurobiError as e:
                if retry_timeout is None:
                    raise ApplicationError(
                        "Could not acquire a Gurobi license: %s" % (e,)
                    ) from e
                if deadline is not None and time.monotonic() >= deadline:
                    raise ApplicationError(
                        "Timed out after %s seconds trying to acquire a Gurobi "
                        "license: %s" % (retry_timeout, e)
                    ) from e
                time.sleep(self._retry_buffer)

    def _create_env(self, timeout: Optional[float] = float("inf")) -> None:
        """Create the shared Gurobi ``Env``"""
        if GurobiDirectBase._gurobipy_env is not None:
            return
        if timeout is not None and timeout != float("inf"):
            try:
                with capture_output(capture_fd=True):
                    env = gurobipy.Env(empty=True)
                    # ServerTimeout is probably the best equivalent to what
                    # we are looking for with our timeout arg. It's Gurobi's
                    # built-in param that controls how long a client waits to
                    # connect to a token server.
                    env.setParam('ServerTimeout', int(timeout))
                    env.start()
                GurobiDirectBase._gurobipy_env = env
                return
            except (gurobipy.GurobiError, AttributeError, TypeError):
                # ServerTimeout not applicable/supported; fall back to the
                # default env creation below (which may still raise, and that
                # will be handled by the retry loop in ``_acquire``).
                pass
        GurobiDirectBase.env()

    def _release(self) -> None:
        GurobiDirectBase._release_env_client()


class GurobiDirectBase(SolverBase):

    _num_gurobipy_env_clients = 0
    _gurobipy_env = None
    _available = None
    _gurobipy_available = gurobipy_available
    _tc_map = None
    _version = None
    _minimum_version = (0, 0, 0)

    CONFIG = GurobiConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.license = _GurobiLicenseManager(self)
        self._callback = None

    def available(
        self,
        recheck: bool = False,
        timeout: Optional[float] = float("inf"),
        retry_timeout: Optional[float] = None,
    ) -> Availability:
        if self._available is not None and not recheck:
            return self._available

        # Note that we set the _available flag on the *most derived
        # class* and not on the instance, or on the base class.  That
        # allows different derived interfaces to have different
        # availability (e.g., persistent has a minimum version
        # requirement that the direct interface doesn't).
        if not self._gurobipy_available:
            self.__class__._available = Availability.NotFound
            return self._available

        # Acquire a license (unless one is already held) to actually probe
        # the license status, then release it if we were the ones who
        # acquired it. If the caller already held a license, we leave it in
        # place. A failed checkout is a reportable status (LicenseError).
        release_on_exit = not self.license.acquired
        try:
            self.license.acquire(timeout=timeout, retry_timeout=retry_timeout)
        except ApplicationError:
            self.__class__._available = Availability.LicenseError
            return self._available
        try:
            self.__class__._available = self._check_license()
            if self.version() < self._minimum_version:
                self.__class__._available = Availability.UnsupportedVersion
        finally:
            if release_on_exit:
                self.license.release()
        return self._available

    @staticmethod
    def release_license():
        if GurobiDirectBase._gurobipy_env is None:
            return
        if GurobiDirectBase._num_gurobipy_env_clients:
            logger.warning(
                "Call to GurobiDirectBase.release_license() with %s remaining "
                "environment clients." % (GurobiDirectBase._num_gurobipy_env_clients,)
            )
        GurobiDirectBase._gurobipy_env.close()
        GurobiDirectBase._gurobipy_env = None

    @staticmethod
    def env():
        if GurobiDirectBase._gurobipy_env is None:
            with capture_output(capture_fd=True):
                GurobiDirectBase._gurobipy_env = gurobipy.Env()
        return GurobiDirectBase._gurobipy_env

    @staticmethod
    def _register_env_client():
        GurobiDirectBase._num_gurobipy_env_clients += 1

    @staticmethod
    def _release_env_client():
        GurobiDirectBase._num_gurobipy_env_clients -= 1
        if GurobiDirectBase._num_gurobipy_env_clients <= 0:
            # Note that _num_gurobipy_env_clients should never be <0,
            # but if it is, release_license will issue a warning (that
            # we want to know about)
            GurobiDirectBase.release_license()

    def _check_license(self):
        try:
            model = gurobipy.Model(env=self.env())
        except gurobipy.GurobiError:
            return Availability.LicenseError

        model.setParam('OutputFlag', 0)
        try:
            model.addVars(range(2001))
            model.optimize()
            return Availability.FullLicense
        except gurobipy.GurobiError:
            return Availability.LimitedLicense
        finally:
            model.dispose()

    def version(self, recheck: bool = False):
        if self._version is None or recheck:
            if not gurobipy_available:
                return None
            self._version = (
                gurobipy.GRB.VERSION_MAJOR,
                gurobipy.GRB.VERSION_MINOR,
                gurobipy.GRB.VERSION_TECHNICAL,
            )
        return self._version

    def _create_solver_model(self, pyomo_model, config):
        # should return gurobi_model, solution_loader, has_objective
        raise NotImplementedError('should be implemented by derived classes')

    def _pyomo_gurobi_var_iter(self):
        # generator of tuples (pyomo_var, gurobi_var)
        raise NotImplementedError('should be implemented by derived classes')

    def _mipstart(self):
        for pyomo_var, gurobi_var in self._pyomo_gurobi_var_iter():
            if pyomo_var.is_integer() and pyomo_var.value is not None:
                gurobi_var.setAttr('Start', pyomo_var.value)

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        tick = time.perf_counter()
        orig_cwd = os.getcwd()
        # Acquire a license for the duration of the solve. If one was already
        # acquired (e.g., the persistent interface holds a license across solves,
        # or the user acquired one explicitly), we reuse it.
        # Otherwise, we release it once the solve is complete. Note that the
        # returned results object (via its solution loader) registers its own
        # environment client, so for the direct interface the underlying
        # Gurobi environment stays alive for as long as the results object is
        # alive.
        release_on_exit = not self.license.acquired
        self.license.acquire()
        try:
            try:
                config = self.config(value=kwds, preserve_implicit=True)

                if config.timer is None:
                    config.timer = HierarchicalTimer()
                timer = config.timer

                StaleFlagManager.mark_all_as_stale()
                ostreams = [io.StringIO()] + config.tee

                if config.working_dir:
                    os.chdir(config.working_dir)
                with capture_output(TeeStream(*ostreams), capture_fd=False):
                    gurobi_model, solution_loader, has_obj = self._create_solver_model(
                        model, config
                    )
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

                    if config.warmstart_discrete_vars:
                        self._mipstart()

                    for key, option in options.items():
                        gurobi_model.setParam(key, option)

                    timer.start('optimize')
                    gurobi_model.optimize(self._callback)
                    timer.stop('optimize')

                res = self._populate_results(
                    grb_model=gurobi_model,
                    solution_loader=solution_loader,
                    has_obj=has_obj,
                    config=config,
                )
            except InfeasibleConstraintException as err:
                err_msg = (
                    'The problem was proven to be infeasible during compilation:\n'
                    f'\t{str(err)}'
                )
                res = get_infeasible_results(
                    model=model, solver=self, config=config, err_msg=err_msg
                )
            finally:
                os.chdir(orig_cwd)
        finally:
            if release_on_exit:
                self.license.release()

        res.solver_log = ostreams[0].getvalue()
        tock = time.perf_counter()
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = tock - tick
        res.timing_info.timer = timer
        return res

    def _get_tc_map(self):
        if GurobiDirectBase._tc_map is None:
            grb = gurobipy.GRB
            tc = TerminationCondition
            GurobiDirectBase._tc_map = {
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
        return GurobiDirectBase._tc_map

    def _populate_results(self, grb_model, solution_loader, has_obj, config):
        status = grb_model.Status

        results = Results()
        results.solution_loader = solution_loader
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

        if has_obj:
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

        config.timer.start('load solution')
        if config.load_solutions:
            if grb_model.SolCount > 0:
                results.solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()
        config.timer.stop('load solution')

        results.solver_config = config
        results.solver_name = self.name
        results.solver_version = self.version()

        return results

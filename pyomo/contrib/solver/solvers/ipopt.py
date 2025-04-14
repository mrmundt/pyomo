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

import logging
import os
import subprocess
import datetime
import io
import re
from typing import Mapping, Optional, Sequence

from pyomo.common import Executable
from pyomo.common.config import (
    ConfigValue,
    document_kwargs_from_configdict,
    ConfigDict,
    ADVANCED_OPTION,
)
from pyomo.common.errors import (
    ApplicationError,
    DeveloperError,
    InfeasibleConstraintException,
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import SolverConfig
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.solvers.sol_reader import parse_sol_file, SolSolutionLoader
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap
from pyomo.solvers.amplfunc_merge import amplfunc_merge

logger = logging.getLogger(__name__)


class IpoptConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.executable: Executable = self.declare(
            'executable',
            ConfigValue(
                default=Executable('ipopt'),
                description="Preferred executable for ipopt. Defaults to searching the "
                "``PATH`` for the first available ``ipopt``.",
            ),
        )
        self.writer_config: ConfigDict = self.declare(
            'writer_config', NLWriter.CONFIG()
        )


class IpoptSolutionLoader(SolSolutionLoader):
    def _error_check(self):
        if self._nl_info is None:
            raise NoSolutionError()
        if len(self._nl_info.eliminated_vars) > 0:
            raise NotImplementedError(
                'For now, turn presolve off (opt.config.writer_config.linear_presolve=False) '
                'to get dual variable values.'
            )
        if self._sol_data is None:
            raise DeveloperError(
                "Solution data is empty. This should not "
                "have happened. Report this error to the Pyomo Developers."
            )

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        self._error_check()
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.variables)
            obj_scale = 1
        else:
            scale_list = self._nl_info.scaling.variables
            obj_scale = self._nl_info.scaling.objectives[0]
        sol_data = self._sol_data
        nl_info = self._nl_info
        zl_map = sol_data.var_suffixes['ipopt_zL_out']
        zu_map = sol_data.var_suffixes['ipopt_zU_out']
        rc = {}
        for ndx, v in enumerate(nl_info.variables):
            scale = scale_list[ndx]
            v_id = id(v)
            rc[v_id] = (v, 0)
            if ndx in zl_map:
                zl = zl_map[ndx] * scale / obj_scale
                if abs(zl) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zl)
            if ndx in zu_map:
                zu = zu_map[ndx] * scale / obj_scale
                if abs(zu) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zu)

        if vars_to_load is None:
            res = ComponentMap(rc.values())
            for v, _ in nl_info.eliminated_vars:
                res[v] = 0
        else:
            res = ComponentMap()
            for v in vars_to_load:
                if id(v) in rc:
                    res[v] = rc[id(v)][1]
                else:
                    # eliminated vars
                    res[v] = 0
        return res


ipopt_command_line_options = {
    'acceptable_compl_inf_tol',
    'acceptable_constr_viol_tol',
    'acceptable_dual_inf_tol',
    'acceptable_tol',
    'alpha_for_y',
    'bound_frac',
    'bound_mult_init_val',
    'bound_push',
    'bound_relax_factor',
    'compl_inf_tol',
    'constr_mult_init_max',
    'constr_viol_tol',
    'diverging_iterates_tol',
    'dual_inf_tol',
    'expect_infeasible_problem',
    'file_print_level',
    'halt_on_ampl_error',
    'hessian_approximation',
    'honor_original_bounds',
    'linear_scaling_on_demand',
    'linear_solver',
    'linear_system_scaling',
    'ma27_pivtol',
    'ma27_pivtolmax',
    'ma57_pivot_order',
    'ma57_pivtol',
    'ma57_pivtolmax',
    'max_cpu_time',
    'max_iter',
    'max_refinement_steps',
    'max_soc',
    'maxit',
    'min_refinement_steps',
    'mu_init',
    'mu_max',
    'mu_oracle',
    'mu_strategy',
    'nlp_scaling_max_gradient',
    'nlp_scaling_method',
    'obj_scaling_factor',
    'option_file_name',
    'outlev',
    'output_file',
    'pardiso_matching_strategy',
    'print_level',
    'print_options_documentation',
    'print_user_options',
    'required_infeasibility_reduction',
    'slack_bound_frac',
    'slack_bound_push',
    'tol',
    'wantsol',
    'warm_start_bound_push',
    'warm_start_init_point',
    'warm_start_mult_bound_push',
    'watchdog_shortened_iter_trigger',
}


class Ipopt(SolverBase):
    CONFIG = IpoptConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._writer = NLWriter()
        self._available_cache = None
        self._version_cache = None
        self._version_timeout = 2

    def available(self, config=None):
        if config is None:
            config = self.config
        pth = config.executable.path()
        if self._available_cache is None or self._available_cache[0] != pth:
            if pth is None:
                self._available_cache = (None, Availability.NotFound)
            else:
                self._available_cache = (pth, Availability.FullLicense)
        return self._available_cache[1]

    def version(self, config=None):
        if config is None:
            config = self.config
        pth = config.executable.path()
        if self._version_cache is None or self._version_cache[0] != pth:
            if pth is None:
                self._version_cache = (None, None)
            else:
                results = subprocess.run(
                    [str(pth), '--version'],
                    timeout=self._version_timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    check=False,
                )
                version = results.stdout.splitlines()[0]
                version = version.split(' ')[1].strip()
                version = tuple(int(i) for i in version.split('.'))
                self._version_cache = (pth, version)
        return self._version_cache[1]

    def has_linear_solver(self, linear_solver):
        import pyomo.core as AML

        m = AML.ConcreteModel()
        m.x = AML.Var()
        m.o = AML.Objective(expr=(m.x - 2) ** 2)
        results = self.solve(
            m,
            tee=False,
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
            solver_options={'linear_solver': linear_solver},
        )
        return 'running with linear solver' in results.solver_log

    def _write_options_file(self, filename: str, options: Mapping):
        # First we need to determine if we even need to create a file.
        # If options is empty, then we return False
        opt_file_exists = False
        if not options:
            return False
        # If it has options in it, parse them and write them to a file.
        # If they are command line options, ignore them; they will be
        # parsed during _create_command_line
        for k, val in options.items():
            if k not in ipopt_command_line_options:
                opt_file_exists = True
                with open(filename + '.opt', 'a+', encoding='utf-8') as opt_file:
                    opt_file.write(str(k) + ' ' + str(val) + '\n')
        return opt_file_exists

    def _create_command_line(self, basename: str, config: IpoptConfig, opt_file: bool):
        cmd = [str(config.executable), basename + '.nl', '-AMPL']
        if opt_file:
            cmd.append('option_file_name=' + basename + '.opt')
        if 'option_file_name' in config.solver_options:
            raise ValueError(
                'Pyomo generates the ipopt options file as part of the `solve` method. '
                'Add all options to ipopt.config.solver_options instead.'
            )
        if (
            config.time_limit is not None
            and 'max_cpu_time' not in config.solver_options
        ):
            config.solver_options['max_cpu_time'] = config.time_limit
        for k, val in config.solver_options.items():
            if k in ipopt_command_line_options:
                cmd.append(str(k) + '=' + str(val))
        return cmd

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds):
        "Solve a model using Ipopt"
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        # Update configuration options, based on keywords passed to solve
        config: IpoptConfig = self.config(value=kwds, preserve_implicit=True)
        # Check if solver is available
        avail = self.available(config)
        if not avail:
            raise ApplicationError(
                f'Solver {self.__class__} is not available ({avail}).'
            )
        if config.threads:
            logger.log(
                logging.WARNING,
                msg="The `threads` option was specified, "
                f"but this is not used by {self.__class__}.",
            )
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        StaleFlagManager.mark_all_as_stale()
        with TempfileManager.new_context() as tempfile:
            if config.working_dir is None:
                dname = tempfile.mkdtemp()
            else:
                dname = config.working_dir
            if not os.path.exists(dname):
                os.mkdir(dname)
            basename = os.path.join(dname, model.name)
            if os.path.exists(basename + '.nl'):
                raise RuntimeError(
                    f"NL file with the same name {basename + '.nl'} already exists!"
                )
            # Note: the ASL has an issue where string constants written
            # to the NL file (e.g. arguments in external functions) MUST
            # be terminated with '\n' regardless of platform.  We will
            # disable universal newlines in the NL file to prevent
            # Python from mapping those '\n' to '\r\n' on Windows.
            with open(
                basename + '.nl', 'w', newline='\n', encoding='utf-8'
            ) as nl_file, open(
                basename + '.row', 'w', encoding='utf-8'
            ) as row_file, open(
                basename + '.col', 'w', encoding='utf-8'
            ) as col_file:
                timer.start('write_nl_file')
                self._writer.config.set_value(config.writer_config)
                try:
                    nl_info = self._writer.write(
                        model,
                        nl_file,
                        row_file,
                        col_file,
                        symbolic_solver_labels=config.symbolic_solver_labels,
                    )
                    proven_infeasible = False
                except InfeasibleConstraintException:
                    proven_infeasible = True
                    nl_info = None
                timer.stop('write_nl_file')

            results = self._solve(proven_infeasible, nl_info, config, timer, basename)

        if (
            config.raise_exception_on_nonoptimal_result
            and results.solution_status != SolutionStatus.optimal
        ):
            raise NoOptimalSolutionError()

        if config.load_solutions:
            if results.solution_status == SolutionStatus.noSolution:
                raise NoFeasibleSolutionError()
            results.solution_loader.load_vars()
            if (
                hasattr(model, 'dual')
                and isinstance(model.dual, Suffix)
                and model.dual.import_enabled()
            ):
                model.dual.update(results.solution_loader.get_duals())
            if (
                hasattr(model, 'rc')
                and isinstance(model.rc, Suffix)
                and model.rc.import_enabled()
            ):
                model.rc.update(results.solution_loader.get_reduced_costs())

        if (
            results.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}
            and len(nl_info.objectives) > 0
        ):
            if config.load_solutions:
                results.incumbent_objective = value(nl_info.objectives[0])
            else:
                results.incumbent_objective = value(
                    replace_expressions(
                        nl_info.objectives[0].expr,
                        substitution_map={
                            id(v): val
                            for v, val in results.solution_loader.get_primals().items()
                        },
                        descend_into_named_expressions=True,
                        remove_named_expressions=True,
                    )
                )

        # Capture/record end-time / wall-time
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (
            end_timestamp - start_timestamp
        ).total_seconds()
        results.timing_info.timer = timer
        return results

    def _solve(self, proven_infeasible, nl_info, config, timer, basename):
        """
        Run the subprocess and generate a Results object with
        relevant information
        """
        try:
            len_nl_info_vars = len(nl_info.variables)
        except AttributeError:
            len_nl_info_vars = None
        if not proven_infeasible and len_nl_info_vars > 0:
            # Get a copy of the environment to pass to the subprocess
            env = os.environ.copy()
            if nl_info.external_function_libraries:
                env['AMPLFUNC'] = amplfunc_merge(
                    env, *nl_info.external_function_libraries
                )
            # Write the opt_file, if there should be one; return a bool to say
            # whether or not we have one (so we can correctly build the command line)
            opt_file = self._write_options_file(
                filename=basename, options=config.solver_options
            )
            # Call ipopt - passing the files via the subprocess
            cmd = self._create_command_line(
                basename=basename, config=config, opt_file=opt_file
            )
            # this seems silly, but we have to give the subprocess slightly
            # longer to finish than ipopt
            if config.time_limit is not None:
                timeout = config.time_limit + min(
                    max(1.0, 0.01 * config.time_limit), 100
                )
            else:
                timeout = None

            ostreams = [io.StringIO()] + config.tee
            with TeeStream(*ostreams) as t:
                timer.start('subprocess')
                process = subprocess.run(
                    cmd,
                    timeout=timeout,
                    env=env,
                    universal_newlines=True,
                    stdout=t.STDOUT,
                    stderr=t.STDERR,
                    check=False,
                )
                timer.stop('subprocess')
                # This is the stuff we need to parse to get the iterations
                # and time
                print(ostreams)
                print(dir(item) for item in ostreams)
                parsed_output_data = self._parse_ipopt_output(ostreams[0])

        if proven_infeasible:
            results = Results()
            results.termination_condition = TerminationCondition.provenInfeasible
            results.solution_loader = SolSolutionLoader(None, None)
            results.iteration_count = 0
            results.timing_info.total_seconds = 0
        elif len_nl_info_vars == 0:
            if len(nl_info.eliminated_vars) == 0:
                results = Results()
                results.termination_condition = TerminationCondition.emptyModel
                results.solution_loader = SolSolutionLoader(None, None)
            else:
                results = Results()
                results.termination_condition = (
                    TerminationCondition.convergenceCriteriaSatisfied
                )
                results.solution_status = SolutionStatus.optimal
                results.solution_loader = SolSolutionLoader(None, nl_info=nl_info)
                results.iteration_count = 0
                results.timing_info.total_seconds = 0
        else:
            if os.path.isfile(basename + '.sol'):
                with open(basename + '.sol', 'r', encoding='utf-8') as sol_file:
                    timer.start('parse_sol')
                    results = self._parse_solution(sol_file, nl_info)
                    timer.stop('parse_sol')
            else:
                results = Results()
            if process.returncode != 0:
                results.extra_info.return_code = process.returncode
                results.termination_condition = TerminationCondition.error
                results.solution_loader = SolSolutionLoader(None, None)
            else:
                results.iteration_count = parsed_output_data['iters']
                parsed_output_data.pop('iters')
                if 'total_time' in parsed_output_data['cpu_seconds']:
                    results.timing_info.total_seconds = parsed_output_data[
                        'cpu_seconds'
                    ]['total_time']
                if 'nofunc_time' in parsed_output_data['cpu_seconds']:
                    results.timing_info.ipopt_excluding_nlp_functions = (
                        parsed_output_data['cpu_seconds']['nofunc_time']
                    )
                    results.timing_info.nlp_function_evaluations = parsed_output_data[
                        'cpu_seconds'
                    ]['func_time']
                parsed_output_data.pop('cpu_seconds')
                results.extra_info = parsed_output_data
                # Set iteration_log visibility to ADVANCED_OPTION because it's
                # a lot to print out with `display`
                results.extra_info.get("iteration_log")._visibility = ADVANCED_OPTION
            if len_nl_info_vars > 0:
                results.solver_log = ostreams[0].getvalue()
        # Populate high-level relevant information
        results.solver_name = self.name
        results.solver_version = self.version(config)
        results.solver_config = config

        return results

    def _parse_ipopt_output(self, output):
        parsed_data = {}

        # Convert output to a string so we can parse it
        if isinstance(output, io.StringIO):
            output = output.getvalue()

        # Extract number of iterations
        iter_match = re.search(r'Number of Iterations\.\.\.\.:\s+(\d+)', output)
        if iter_match:
            parsed_data['iters'] = int(iter_match.group(1))

        iter_table = re.findall(r'^(?:\s*\d+.*?)$', output, re.MULTILINE)
        if iter_table:
            columns = [
                "iter",
                "objective",
                "inf_pr",
                "inf_du",
                "lg_mu",
                "d_norm",
                "lg_rg",
                "alpha_du",
                "alpha_pr",
                "ls",
            ]
            all_iterations = []

            for line in iter_table:
                tokens = line.strip().split()
                if len(tokens) == len(columns):
                    iter_data = dict(zip(columns, tokens))

                    # Extract restoration flag from 'iter'
                    iter_val = iter_data["iter"]
                    iter_match = re.match(r"(\d+)(r?)", iter_val)
                    if iter_match:
                        iter_data["restoration"] = bool(
                            iter_match.group(2)
                        )  # True if 'r' is present
                    else:
                        iter_data["restoration"] = False
                    iter_data.pop('iter')

                    # Separate alpha_pr into numeric part and optional tag
                    alpha_pr_val = iter_data["alpha_pr"]
                    match = re.match(r"([0-9.eE+-]+)([a-zA-Z]?)", alpha_pr_val)
                    if match:
                        iter_data["alpha_pr"] = match.group(1)
                        iter_data["step_acceptance"] = (
                            match.group(2) if match.group(2) else None
                        )
                    else:
                        iter_data["step_acceptance"] = None

                    # Attempt to cast all values to float where possible
                    for key in iter_data:
                        try:
                            if iter_data[key] == '-':
                                iter_data[key] = None
                                continue
                            if isinstance(iter_data[key], bool):
                                continue
                            iter_data[key] = float(iter_data[key])
                        except (ValueError, TypeError):
                            pass

                    all_iterations.append(iter_data)
            parsed_data['iteration_log'] = all_iterations

        # Extract scaled and unscaled table
        scaled_unscaled_match = re.findall(
            r'Objective\.\.+:\s+([-+eE0-9.]+)\s+([-+eE0-9.]+).*?'
            r'Dual infeasibility\.\.+:\s+([-+eE0-9.]+)\s+([-+eE0-9.]+).*?'
            r'Constraint violation\.\.+:\s+([-+eE0-9.]+)\s+([-+eE0-9.]+).*?'
            r'Complementarity\.\.+:\s+([-+eE0-9.]+)\s+([-+eE0-9.]+).*?'
            r'Overall NLP error\.\.+:\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)',
            output,
            re.DOTALL,
        )
        if scaled_unscaled_match:
            scaled = {
                "incumbent_objective": float(scaled_unscaled_match[0][0]),
                "dual_infeasibility": float(scaled_unscaled_match[0][2]),
                "constraint_violation": float(scaled_unscaled_match[0][4]),
                "complementarity_error": float(scaled_unscaled_match[0][6]),
                "overall_nlp_error": float(scaled_unscaled_match[0][8]),
            }
            unscaled = {
                "incumbent_objective": float(scaled_unscaled_match[0][1]),
                "dual_infeasibility": float(scaled_unscaled_match[0][3]),
                "constraint_violation": float(scaled_unscaled_match[0][5]),
                "complementarity": float(scaled_unscaled_match[0][7]),
                "overall_nlp_error": float(scaled_unscaled_match[0][9]),
            }

            parsed_data.update(unscaled)
            parsed_data['final_scaled_results'] = scaled

        # Newer versions of IPOPT no longer separate timing into
        # two different values. This is so we have compatibility with
        # both new and old versions
        cpu_match = re.findall(r'Total CPU secs in .*? =\s+([0-9.]+)', output)
        ipopt_seconds_match = re.search(
            r'Total seconds in IPOPT\s+=\s+([0-9.]+)', output
        )
        if cpu_match and len(cpu_match) >= 2:
            parsed_data['cpu_seconds'] = {
                'nofunc_time': float(cpu_match[0]),
                'func_time': float(cpu_match[1]),
            }
        elif ipopt_seconds_match:
            parsed_data['cpu_seconds'] = {
                'total_time': float(ipopt_seconds_match.group(1))
            }

        return parsed_data

    def _parse_solution(self, instream: io.TextIOBase, nl_info: NLWriterInfo):
        results = Results()
        res, sol_data = parse_sol_file(
            sol_file=instream, nl_info=nl_info, result=results
        )

        if res.solution_status == SolutionStatus.noSolution:
            res.solution_loader = SolSolutionLoader(None, None)
        else:
            res.solution_loader = IpoptSolutionLoader(
                sol_data=sol_data, nl_info=nl_info
            )

        return res

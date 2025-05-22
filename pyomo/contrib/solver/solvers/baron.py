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
import datetime
import os
import io
import subprocess
import re
import sys
from typing import Optional, Tuple

from pyomo.common import Executable
from pyomo.common.config import ConfigValue, document_kwargs_from_configdict, ConfigDict
from pyomo.common.errors import (
    ApplicationError,
    DeveloperError,
    InfeasibleConstraintException,
)
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
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

logger = logging.getLogger(__name__)

"""
NOTES

  - Baron has its own file type (`bar`)
  - You can't run `./baron --version`; you need a dummy input file
  - License status can be checked by passing in something with 11 vars (<=10 is demo mode)
  - Baron takes the input of `bar` and outputs (at least) two files: res.lst and tim.lst
    - res.lst - results, has the actual solution stuff
    - tim.lst - supposedly has timing information?

"""


class BaronConfig(SolverConfig):
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
                default=Executable('baron'),
                description="Preferred executable for BARON. Defaults to searching the "
                "``PATH`` for the first available ``baron``.",
            ),
        )


class Baron(SolverBase):
    CONFIG = BaronConfig()

    def __init__(self, **kwds) -> None:
        super().__init__(**kwds)
        self._writer = None
        self._available_cache = None
        self._version_cache = None
        self._version_timeout = 2
        self._license_cache = None

    def available(self) -> Availability:
        if self._available_cache is None:
            pth = self.config.executable.path()
            if self._available_cache is None or self._available_cache[0] != pth:
                if pth is None:
                    self._available_cache = (None, Availability.NotFound)
                else:
                    self._available_cache = (pth, Availability.Installed)
            return self._available_cache[1]

    def version(self) -> Optional[Tuple[int, int, int]]:
        pth = self.config.executable.path()
        if self._version_cache is None or self._version_cache[0] != pth:
            if pth is None:
                self._version_cache = (None, None)
            else:
                with TempfileManager.new_context() as tempfile:
                    # Create a dummy bar file so we can check version
                    with open(
                        os.path.join(tempfile.mkdtemp(), 'version.bar'), 'w'
                    ) as f:
                        options_block = f"""//This is a dummy .bar file created to return the baron version//
                                        OPTIONS {{
                                        results: 1;
                                        ResName: "{f.name}";
                                        summary: 1;
                                        SumName: "{f.name}";
                                        times: 1;
                                        TimName: "{f.name}";
                                        }}
                                        POSITIVE_VARIABLES x1;
                                        OBJ: minimize x1;"""
                        f.write(options_block)
                    results = subprocess.run(
                        [str(pth), f.name],
                        timeout=self._version_timeout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        check=False,
                    )
                    match = re.search(
                        r'BARON version (\d+)\.(\d+)\.(\d+)', results.stdout
                    )
                    if match:
                        self._version_cache = (pth, tuple(map(int, match.groups())))
                    else:
                        self._version_cache = (None, None)
                        logger.warning(
                            f"BARON (from {str(pth)}) returned an error code: {results.returncode}.\n"
                            f"Captured output: {results.stdout}"
                        )
            return self._version_cache[1]

    def license_is_valid(self) -> Optional[bool]:
        pth = self.config.executable.path()
        if self._license_cache is None or self._license_cache[0] != pth:
            if pth is None:
                self._license_cache = (None, None)
            elif self.available():
                with TempfileManager.new_context() as tempfile:
                    # Create a dummy bar file so we can check license validity
                    with open(
                        os.path.join(tempfile.mkdtemp(), 'license.bar'), 'w'
                    ) as f:
                        options_block = f"""//This is a dummy .bar file created to return the baron version//
                                        OPTIONS {{
                                        results: 1;
                                        ResName: "{f.name}";
                                        summary: 1;
                                        SumName: "{f.name}";
                                        times: 1;
                                        TimName: "{f.name}";
                                        }}
                                        POSITIVE_VARIABLES {" ,".join("x" + str(i) for i in range(11))};
                                        OBJ: minimize x1;"""
                        f.write(options_block)
                    results = subprocess.run(
                        [str(pth), f.name],
                        timeout=self._version_timeout,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        check=False,
                    )
                if results.returncode == 0:
                    match = re.search('Licensing error', results.stdout)
                    # We also want to update the license information
                    # This may need to be reworked per Pyomo/pyomo#3516
                    if match:
                        self._license_cache = (pth, False)
                        self._available_cache = (pth, Availability.LimitedLicense)
                    else:
                        self._license_cache = (pth, True)
                        self._available_cache = (pth, Availability.FullLicense)
                else:
                    self._license_cache = (None, None)
                    logger.warning(
                        f"BARON (from {str(pth)}) returned an error code {results.returncode}.\n"
                        f"Captured output: {results.stdout}"
                    )
            else:
                self._license_cache = (None, None)
                logger.warning(f"BARON (from {str(pth)}) is not available.")
            return self._license_cache[1]

    @document_kwargs_from_configdict(CONFIG)
    def solve(self, model, **kwds) -> Results:
        # Begin time tracking
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        # Update configuration options, based on keywords passed to solve
        config: BaronConfig = self.config(value=kwds, preserve_implicit=True)
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
        with TempfileManager.new_context() as tempfile:
            if config.working_dir is None:
                dname = tempfile.mkdtemp()
            else:
                dname = config.working_dir
            if not os.path.exists(dname):
                os.mkdir(dname)
            basename = os.path.join(dname, model.name)
            if os.path.exists(basename + '.bar'):
                raise RuntimeError(
                    f"BARON file with the same name {basename + '.bar'} already exists!"
                )
            with open(basename + '.bar', 'w') as f:
                # Not sure what to do after this point. There isn't a nice writer
                # interface for BARON like there is for nl on ipopt. I'm guessing
                # this is in the backlog?
                pass
            # this seems silly, but we have to give the subprocess slightly
            # longer to finish than baron
            if config.time_limit is not None:
                timeout = config.time_limit + min(
                    max(1.0, 0.01 * config.time_limit), 100
                )
            else:
                timeout = None

            ostreams = [io.StringIO()] + config.tee
            timer.start('subprocess')
            cmd = [str(config.executable), f.name]
            try:
                with TeeStream(*ostreams) as t:
                    process = subprocess.run(
                        cmd,
                        timeout=timeout,
                        universal_newlines=True,
                        stdout=t.STDOUT,
                        stderr=t.STDERR,
                        check=False,
                    )
            except OSError:
                err = sys.exc_info()[1]
                msg = 'Could not execute the command: %s\tError message: %s'
                raise ApplicationError(msg % (cmd, err))
            finally:
                timer.stop('subprocess')

    def _parse_time_file(self, filename):
        "Parse the tim.lst file"

    def _parse_soln_file(self, filename):
        "Parse the res.lst file"

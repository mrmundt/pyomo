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
import subprocess
import re
from typing import Mapping, Optional, Sequence

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

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._available_cache = None
        self._version_cache = None
        self._version_timeout = 2
        self._license_cache = None

    def available(self):
        if self._available_cache is None:
            pth = self.config.executable.path()
            if self._available_cache is None or self._available_cache[0] != pth:
                if pth is None:
                    self._available_cache = (None, Availability.NotFound)
                else:
                    self._available_cache = (pth, Availability.Installed)
            return self._available_cache[1]

    def version(self):
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
                    if results.returncode == 0:
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

    def license_is_valid(self):
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

    def solve(self):
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)

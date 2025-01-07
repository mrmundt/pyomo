#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# The log should be imported first so that the Pyomo LogHandler can be
# set up as soon as possible
from pyomo.common import log
from pyomo.common import envvar

from pyomo.common.factory import Factory

from pyomo.common.fileutils import (
    Executable,
    Library,
    # The following will be deprecated soon
    register_executable,
    registered_executable,
    unregister_executable,
)
from pyomo.common import config, dependencies, shutdown, timing
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import DeveloperError
from pyomo.common._command import pyomo_command, get_pyomo_commands

#
# declare deprecation paths for removed modules
#
from pyomo.common.deprecation import moved_module

moved_module('pyomo.common.getGSL', 'pyomo.common.gsl', version='6.5.0')
moved_module('pyomo.common.plugin', 'pyomo.common.plugin_base', version='6.5.0')
del moved_module

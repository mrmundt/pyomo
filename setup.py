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

"""
Script to generate the installer for pyomo.
"""

import os
import platform
import sys
from setuptools import setup

def check_config_arg(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    if name in os.getenv('PYOMO_SETUP_ARGS', '').split():
        return True
    return False

def import_pyomo_module(*path):
    _module_globals = dict(globals())
    _module_globals['__name__'] = None
    _source = os.path.join(os.path.dirname(__file__), *path)
    with open(_source) as _FILE:
        exec(_FILE.read(), _module_globals)
    return _module_globals

# Handle Cython extensions if requested
ext_modules = []
CYTHON_REQUIRED = "required"
using_cython = False

if not any(
    arg.startswith(cmd)
    for cmd in ('build', 'install', 'bdist', 'wheel')
    for arg in sys.argv
):
    using_cython = False
elif sys.version_info[:2] < (3, 11):
    using_cython = "automatic"
else:
    using_cython = False
if check_config_arg('--with-cython'):
    using_cython = CYTHON_REQUIRED
if check_config_arg('--without-cython'):
    using_cython = False

if using_cython:
    try:
        if platform.python_implementation() != "CPython":
            raise RuntimeError("Cython is only supported under CPython")
        from Cython.Build import cythonize
        import shutil

        #
        # Note: The Cython developers recommend that you distribute C source
        # files to users.  But this is fine for evaluating the utility of Cython
        #
        files = [
            "pyomo/core/expr/numvalue.pyx",
            "pyomo/core/expr/numeric_expr.pyx",
            "pyomo/core/expr/logical_expr.pyx",
            # "pyomo/core/expr/visitor.pyx",
            "pyomo/core/util.pyx",
            "pyomo/repn/standard_repn.pyx",
            "pyomo/repn/plugins/cpxlp.pyx",
            "pyomo/repn/plugins/gams_writer.pyx",
            "pyomo/repn/plugins/baron_writer.pyx",
            "pyomo/repn/plugins/ampl/ampl_.pyx",
        ]
        for f in files:
            shutil.copyfile(f[:-1], f)
        ext_modules = cythonize(files, compiler_directives={"language_level": 3})
    except:
        if using_cython == CYTHON_REQUIRED:
            print(
                """
ERROR: Cython was explicitly requested with --with-cython, but cythonization
       of core Pyomo modules failed.
"""
            )
            raise
        using_cython = False

if check_config_arg('--with-distributable-extensions'):
    #
    # Import the APPSI extension builder
    # NOTE: There is inconsistent behavior in Windows for APPSI.
    # As a result, we will NOT include these extensions in Windows.
    if not sys.platform.startswith('win'):
        appsi_extension = import_pyomo_module('pyomo', 'contrib', 'appsi', 'build.py')[
            'get_appsi_extension'
        ](
            in_setup=True,
            appsi_root=os.path.join(
                os.path.dirname(__file__), 'pyomo', 'contrib', 'appsi'
            ),
        )
        ext_modules.append(appsi_extension)

# Most metadata is now in pyproject.toml, but we will continue to handle
# the Cython extensions here. pyproject.toml support is not yet up to par.
setup_kwargs = dict(
    ext_modules=ext_modules,
)

try:
    setup(**setup_kwargs)
except SystemExit as e_info:
    # Cython can generate a SystemExit exception on Windows if the
    # environment is missing / has an incorrect Microsoft compiler.
    # Since Cython is not strictly required, we will disable Cython and
    # try re-running setup(), but only for this very specific situation.
    if 'Microsoft Visual C++' not in str(e_info):
        raise
    elif using_cython == CYTHON_REQUIRED:
        print(
            """
ERROR: Cython was explicitly requested with --with-cython, but cythonization
   of core Pyomo modules failed.
"""
        )
        raise
    else:
        print(
            """
ERROR: setup() failed:
%s
Re-running setup() without the Cython modules
"""
            % (str(e_info),)
        )
        setup_kwargs['ext_modules'] = []
        setup(**setup_kwargs)
        print(
            """
WARNING: Installation completed successfully, but the attempt to cythonize
     core Pyomo modules failed.  Cython provides performance
     optimizations and is not required for any Pyomo functionality.
     Cython returned the following error:
   "%s"
"""
            % (str(e_info),)
        )

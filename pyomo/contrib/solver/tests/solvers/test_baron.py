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

import os
import subprocess

import pyomo.environ as pyo
import pyomo.contrib.solver.solvers.baron as baron
from pyomo.common import unittest, Executable

baron_available = baron.Baron().available()


@unittest.skipIf(not baron_available, "The 'baron' command is not available")
class TestBaronInterface(unittest.TestCase):
    def test_version_cache(self):
        opt = baron.Baron()
        opt.version()
        self.assertIsNotNone(opt._version_cache[0])
        self.assertIsNotNone(opt._version_cache[1])
        # Now we will try with a custom config that has a fake path
        opt.config.executable = Executable('/a/bogus/path')
        opt.version()
        self.assertIsNone(opt._version_cache[0])
        self.assertIsNone(opt._version_cache[1])

    def test_license_cache(self):
        opt = baron.Baron()
        opt.license_is_valid()
        self.assertIsNotNone(opt._license_cache[0])
        self.assertIsNotNone(opt._license_cache[1])
        # Now we will try with a custom config that has a fake path
        opt.config.executable = Executable('/a/bogus/path')
        opt.license_is_valid()
        self.assertIsNone(opt._license_cache[0])
        self.assertIsNone(opt._license_cache[1])

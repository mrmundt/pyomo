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

from pyomo.common import unittest
from pyomo.solvers.plugins.solvers import IPOPT
from pyomo.common.tee import capture_output
import pyomo.environ as pyo

ipopt_available = IPOPT.IPOPT().available()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):
    def test_has_linear_solver(self):
        opt = IPOPT.IPOPT()
        self.assertTrue(
            any(
                map(
                    opt.has_linear_solver,
                    [
                        'mumps',
                        'ma27',
                        'ma57',
                        'ma77',
                        'ma86',
                        'ma97',
                        'pardiso',
                        'pardisomkl',
                        'spral',
                        'wsmp',
                    ],
                )
            )
        )
        self.assertFalse(opt.has_linear_solver('bogus_linear_solver'))


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpopt(unittest.TestCase):
    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5)
        model.y = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        return model

    def test_ipopt_tee_true(self):
        model = self.create_model()
        with capture_output() as OUT:
            result = IPOPT.IPOPT().solve(model, tee=True)
            output = OUT.getvalue()
        print(output)
        self.assertIn("Optimal Solution Found", output)

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import gc

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct import GurobiDirect
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_base import GurobiDirectBase

opt = GurobiDirect()
if not opt.available():
    raise unittest.SkipTest("gurobi is not available")


def create_lp_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(bounds=(0, 10))
    m.c = pyo.Constraint(expr=m.x + m.y >= 5)
    m.obj = pyo.Objective(expr=2 * m.x + 3 * m.y)
    return m


@unittest.pytest.mark.solver("gurobi_direct")
class TestGurobiDirectInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = GurobiDirect()
        expected_members = {
            'CONFIG',
            'available',
            'config',
            'api_version',
            'is_persistent',
            'license',
            'name',
            'release_license',
            'env',
            'solve',
            'version',
        }
        method_list = {method for method in dir(opt) if not method.startswith('_')}
        self.assertTrue(expected_members.issubset(method_list))

    def test_default_instantiation(self):
        opt = GurobiDirect()
        self.assertFalse(opt.is_persistent())
        self.assertIn(
            opt.available(),
            {
                Availability.NotFound,
                Availability.UnsupportedVersion,
                Availability.LimitedLicense,
                Availability.FullLicense,
                Availability.LicenseError,
            },
        )

    def test_version(self):
        opt = GurobiDirect()
        ver = opt.version()
        self.assertIsInstance(ver, tuple)
        self.assertGreaterEqual(len(ver), 3)
        self.assertTrue(all(isinstance(_, int) for _ in ver))


@unittest.pytest.mark.solver("gurobi_direct")
class TestGurobiDirectLicense(unittest.TestCase):
    def setUp(self):
        gc.collect()

    def test_available_no_license(self):
        opt = GurobiDirect()
        # Constructing a solver should not check out a license
        self.assertFalse(opt.license.acquired)
        result = opt.available(recheck=True)
        self.assertTrue(bool(result))
        # available should release the license it acquired
        self.assertFalse(opt.license.acquired)

    def test_available_held_license(self):
        opt = GurobiDirect()
        opt.license.acquire()
        try:
            self.assertTrue(opt.license.acquired)
            opt.available(recheck=True)
            # available should leave a pre-held license in place
            self.assertTrue(opt.license.acquired)
        finally:
            opt.license.release()
        self.assertFalse(opt.license.acquired)

    def test_double_acquire(self):
        opt = GurobiDirect()
        opt.license.acquire()
        clients = GurobiDirectBase._num_gurobipy_env_clients
        # A second acquire shouldn't check out another client
        opt.license.acquire()
        self.assertEqual(clients, GurobiDirectBase._num_gurobipy_env_clients)
        opt.license.release()
        self.assertFalse(opt.license.acquired)

    def test_context_manager(self):
        opt = GurobiDirect()
        self.assertFalse(opt.license.acquired)
        with opt.license:
            self.assertTrue(opt.license.acquired)
        self.assertFalse(opt.license.acquired)

    def test_solve_env(self):
        m = create_lp_model()
        opt = GurobiDirect()
        res = opt.solve(m)
        self.assertFalse(opt.license.acquired)
        duals = res.solution_loader.get_duals()
        self.assertIn(m.c, duals)

    def test_env_removed(self):
        m = create_lp_model()
        opt = GurobiDirect()
        res = opt.solve(m)
        self.assertFalse(opt.license.acquired)
        # Removing the last reference to the results object (and its solution
        # loader) should release the remaining environment client and clear
        # the license
        del res
        gc.collect()
        self.assertLessEqual(GurobiDirectBase._num_gurobipy_env_clients, 0)


if __name__ == '__main__':
    unittest.main()

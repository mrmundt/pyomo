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

import abc
from typing import Sequence, Dict, Optional, Mapping, NoReturn

from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.sol_reader import SolFileData
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.core.expr.visitor import replace_expressions


class SolutionLoaderBase(abc.ABC):
    """
    Base class for all future SolutionLoader classes.

    Intent of this class and its children is to load the solution back into the model.
    """

    def load_vars(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> NoReturn:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            The minimum set of variables whose solution should be loaded. If vars_to_load is None, then the solution
            to all primal variables will be loaded. Even if vars_to_load is specified, the values of other
            variables may also be loaded depending on the interface.
        """
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    @abc.abstractmethod
    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        """
        Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution value should be retrieved. If vars_to_load is None,
            then the values for all variables will be retrieved.

        Returns
        -------
        primals: ComponentMap
            Maps variables to solution values
        """

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        """
        Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be retrieved. If cons_to_load is None, then the duals for all
            constraints will be retrieved.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        """
        Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be retrieved. If vars_to_load is None, then the
            reduced costs for all variables will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variables to reduced costs
        """
        raise NotImplementedError(
            f'{type(self)} does not support the get_reduced_costs method'
        )


class PersistentSolutionLoader(SolutionLoaderBase):
    def __init__(self, solver):
        self._solver = solver
        self._valid = True

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver._get_primals(vars_to_load=vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_duals(cons_to_load=cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        self._assert_solution_still_valid()
        return self._solver._get_reduced_costs(vars_to_load=vars_to_load)

    def invalidate(self):
        self._valid = False


class SolSolutionLoader(SolutionLoaderBase):
    def __init__(self, sol_data: SolFileData, nl_info: NLWriterInfo) -> None:
        self._sol_data = sol_data
        self._nl_info = nl_info

    def load_vars(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> NoReturn:
        if self._nl_info.scaling:
            for v, val, scale in zip(
                self._nl_info.variables, self._sol_data.primals, self._nl_info.scaling
            ):
                v.set_value(val / scale, skip_validation=True)
        else:
            for v, val in zip(self._nl_info.variables, self._sol_data.primals):
                v.set_value(val, skip_validation=True)

        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.variables)
        else:
            scale_list = self._nl_info.scaling.variables
        val_map = dict()
        for v, val, scale in zip(
            self._nl_info.variables, self._sol_data.primals, scale_list
        ):
            val_map[id(v)] = val / scale

        for v, v_expr in self._nl_info.eliminated_vars:
            val = replace_expressions(v_expr, substitution_map=val_map)
            v_id = id(v)
            val_map[v_id] = val

        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._nl_info.variables + [
                v for v, _ in self._nl_info.eliminated_vars
            ]
        for v in vars_to_load:
            res[v] = val_map[id(v)]

        return res

    def get_duals(
        self, cons_to_load: Optional[Sequence[_GeneralConstraintData]] = None
    ) -> Dict[_GeneralConstraintData, float]:
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.constraints)
        else:
            scale_list = self._nl_info.scaling.constraints
        if cons_to_load is None:
            cons_to_load = set(self._nl_info.constraints)
        else:
            cons_to_load = set(cons_to_load)
        res = dict()
        for c, val, scale in zip(
            self._nl_info.constraints, self._sol_data.duals, scale_list
        ):
            if c in cons_to_load:
                res[c] = val * scale
        return res
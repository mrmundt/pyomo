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
'''
Example from Y. Bard, "Nonlinear Parameter Estimation", (pg. 124)

This example shows:
1. How to define the unknown (to be regressed parameters) with an index
2. How to call parmest to only estimate some of the parameters (and fix the rest)

Code provided by Paul Akula.
'''

import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel,
    Param,
    Var,
    PositiveReals,
    Objective,
    Constraint,
    RangeSet,
    Expression,
    minimize,
    exp,
    value,
)
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment


def simple_reaction_model(data):
    # Create the concrete model
    model = ConcreteModel()

    model.x1 = Param(initialize=float(data['x1']))
    model.x2 = Param(initialize=float(data['x2']))

    # Rate constants
    model.rxn = RangeSet(2)
    initial_guess = {1: 750, 2: 1200}
    model.k = Var(model.rxn, initialize=initial_guess, within=PositiveReals)

    # reaction product
    model.y = Expression(expr=exp(-model.k[1] * model.x1 * exp(-model.k[2] / model.x2)))

    # fix all of the regressed parameters
    model.k.fix()

    # ===================================================================
    # Stage-specific cost computations
    def ComputeFirstStageCost_rule(model):
        return 0

    model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

    def AllMeasurements(m):
        return (float(data['y']) - m.y) ** 2

    model.SecondStageCost = Expression(rule=AllMeasurements)

    def total_cost_rule(m):
        return m.FirstStageCost + m.SecondStageCost

    model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)

    return model


# For this experiment class, data is dictionary
class SimpleReactionExperiment(Experiment):

    def __init__(self, data):
        self.data = data
        self.model = None

    def create_model(self):
        self.model = simple_reaction_model(self.data)

    def label_model(self):

        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update(
            [(m.x1, self.data['x1']), (m.x2, self.data['x2']), (m.y, self.data['y'])]
        )

        return m

    def get_labeled_model(self):
        self.create_model()
        m = self.label_model()

        return m


# k[2] fixed
class SimpleReactionExperimentK2Fixed(SimpleReactionExperiment):

    def label_model(self):

        m = super().label_model()

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.k[1]])

        return m


# k[2] variable
class SimpleReactionExperimentK2Variable(SimpleReactionExperiment):

    def label_model(self):

        m = super().label_model()

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.k[1], m.k[2]])

        return m


def main():

    # Data from Table 5.2 in  Y. Bard, "Nonlinear Parameter Estimation", (pg. 124)
    data = [
        {'experiment': 1, 'x1': 0.1, 'x2': 100, 'y': 0.98},
        {'experiment': 2, 'x1': 0.2, 'x2': 100, 'y': 0.983},
        {'experiment': 3, 'x1': 0.3, 'x2': 100, 'y': 0.955},
        {'experiment': 4, 'x1': 0.4, 'x2': 100, 'y': 0.979},
        {'experiment': 5, 'x1': 0.5, 'x2': 100, 'y': 0.993},
        {'experiment': 6, 'x1': 0.05, 'x2': 200, 'y': 0.626},
        {'experiment': 7, 'x1': 0.1, 'x2': 200, 'y': 0.544},
        {'experiment': 8, 'x1': 0.15, 'x2': 200, 'y': 0.455},
        {'experiment': 9, 'x1': 0.2, 'x2': 200, 'y': 0.225},
        {'experiment': 10, 'x1': 0.25, 'x2': 200, 'y': 0.167},
        {'experiment': 11, 'x1': 0.02, 'x2': 300, 'y': 0.566},
        {'experiment': 12, 'x1': 0.04, 'x2': 300, 'y': 0.317},
        {'experiment': 13, 'x1': 0.06, 'x2': 300, 'y': 0.034},
        {'experiment': 14, 'x1': 0.08, 'x2': 300, 'y': 0.016},
        {'experiment': 15, 'x1': 0.1, 'x2': 300, 'y': 0.006},
    ]

    # Create an experiment list with k[2] fixed
    exp_list = []
    for i in range(len(data)):
        exp_list.append(SimpleReactionExperimentK2Fixed(data[i]))

    # View one model
    # exp0_model = exp_list[0].get_labeled_model()
    # exp0_model.pprint()

    # =======================================================================
    # Parameter estimation without covariance estimate
    # Only estimate the parameter k[1]. The parameter k[2] will remain fixed
    # at its initial value

    pest = parmest.Estimator(exp_list)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)
    print()

    # Create an experiment list with k[2] variable
    exp_list = []
    for i in range(len(data)):
        exp_list.append(SimpleReactionExperimentK2Variable(data[i]))

    # =======================================================================
    # Estimate both k1 and k2 and compute the covariance matrix
    pest = parmest.Estimator(exp_list)
    n = 15  # total number of data points used in the objective (y in 15 scenarios)
    obj, theta, cov = pest.theta_est(calc_cov=True, cov_n=n)
    print(obj)
    print(theta)
    print(cov)


if __name__ == "__main__":
    main()

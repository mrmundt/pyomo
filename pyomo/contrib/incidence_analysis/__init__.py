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

from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface, get_bipartite_incidence_graph
from pyomo.contrib.incidence_analysis.scc_solver import (
    generate_strongly_connected_components,
    solve_strongly_connected_components,
)
from pyomo.contrib.incidence_analysis.incidence import get_incident_variables
from pyomo.contrib.incidence_analysis.config import IncidenceMethod

#
# declare deprecation paths for removed modules
#
from pyomo.common.deprecation import moved_module

moved_module(
    "pyomo.contrib.incidence_analysis.util",
    "pyomo.contrib.incidence_analysis.scc_solver",
    version='6.5.0',
    msg=(
        "The 'pyomo.contrib.incidence_analysis.util' module has been moved to "
        "'pyomo.contrib.incidence_analysis.scc_solver'. However, we recommend "
        "importing this functionality (e.g. solve_strongly_connected_components) "
        "directly from 'pyomo.contrib.incidence_analysis'."
    ),
)
del moved_module

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
import operator
from math import prod as _prod

import pyomo.core.expr as EXPR
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
from pyomo.core.expr.numvalue import value, native_types

#
# Sympy takes a significant time to load; defer importing it unless
# someone actually needs the interface.
#

_operatorMap = {}
_pyomo_operator_map = {}
_functionMap = {}


def _nondifferentiable(x):
    if type(x[1]) is tuple:
        # sympy >= 1.3 returns tuples (var, order)
        wrt = x[1][0]
    else:
        # early versions of sympy returned the bare var
        wrt = x[1]
    raise NondifferentiableError(
        "The sub-expression '%s' is not differentiable with respect to %s" % (x[0], wrt)
    )


def _external_fcn(*x):
    raise TypeError(
        "Expressions containing external functions are not convertible to "
        f"sympy expressions (found 'f{x}')"
    )


def _configure_sympy(sympy, available):
    if not available:
        return

    _operatorMap.update(
        {
            sympy.Add: sum,
            sympy.Mul: _prod,
            sympy.Pow: lambda x: operator.pow(*x),
            sympy.exp: lambda x: EXPR.exp(*x),
            sympy.log: lambda x: EXPR.log(*x),
            sympy.sin: lambda x: EXPR.sin(*x),
            sympy.asin: lambda x: EXPR.asin(*x),
            sympy.sinh: lambda x: EXPR.sinh(*x),
            sympy.asinh: lambda x: EXPR.asinh(*x),
            sympy.cos: lambda x: EXPR.cos(*x),
            sympy.acos: lambda x: EXPR.acos(*x),
            sympy.cosh: lambda x: EXPR.cosh(*x),
            sympy.acosh: lambda x: EXPR.acosh(*x),
            sympy.tan: lambda x: EXPR.tan(*x),
            sympy.atan: lambda x: EXPR.atan(*x),
            sympy.tanh: lambda x: EXPR.tanh(*x),
            sympy.atanh: lambda x: EXPR.atanh(*x),
            sympy.ceiling: lambda x: EXPR.ceil(*x),
            sympy.floor: lambda x: EXPR.floor(*x),
            sympy.sqrt: lambda x: EXPR.sqrt(*x),
            sympy.Abs: lambda x: abs(*x),
            sympy.Derivative: _nondifferentiable,
            sympy.Tuple: lambda x: x,
            sympy.Or: lambda x: EXPR.lor(*x),
            sympy.And: lambda x: EXPR.land(*x),
            sympy.Implies: lambda x: EXPR.implies(*x),
            sympy.Equivalent: lambda x: EXPR.equivalents(*x),
            sympy.Not: lambda x: EXPR.lnot(*x),
            sympy.LessThan: lambda x: operator.le(*x),
            sympy.StrictLessThan: lambda x: operator.lt(*x),
            sympy.GreaterThan: lambda x: operator.ge(*x),
            sympy.StrictGreaterThan: lambda x: operator.gt(*x),
            sympy.Equality: lambda x: operator.eq(*x),
        }
    )

    _pyomo_operator_map.update(
        {
            EXPR.SumExpression: sympy.Add,
            EXPR.LinearExpression: sympy.Add,
            EXPR.ProductExpression: sympy.Mul,
            EXPR.MonomialTermExpression: sympy.Mul,
            EXPR.ExternalFunctionExpression: _external_fcn,
            EXPR.AndExpression: sympy.And,
            EXPR.OrExpression: sympy.Or,
            EXPR.ImplicationExpression: sympy.Implies,
            EXPR.EquivalenceExpression: sympy.Equivalent,
            EXPR.XorExpression: sympy.Xor,
            EXPR.NotExpression: sympy.Not,
        }
    )

    _functionMap.update(
        {
            'exp': sympy.exp,
            'log': sympy.log,
            'log10': lambda x: sympy.log(x) / sympy.log(10),
            'sin': sympy.sin,
            'asin': sympy.asin,
            'sinh': sympy.sinh,
            'asinh': sympy.asinh,
            'cos': sympy.cos,
            'acos': sympy.acos,
            'cosh': sympy.cosh,
            'acosh': sympy.acosh,
            'tan': sympy.tan,
            'atan': sympy.atan,
            'tanh': sympy.tanh,
            'atanh': sympy.atanh,
            'ceil': sympy.ceiling,
            'floor': sympy.floor,
            'sqrt': sympy.sqrt,
        }
    )


sympy, sympy_available = attempt_import('sympy', callback=_configure_sympy)


class PyomoSympyBimap(object):
    def __init__(self):
        self.pyomo2sympy = ComponentMap()
        self.sympy2pyomo = {}
        self.i = 0

    def getPyomoSymbol(self, sympy_object, default=None):
        return self.sympy2pyomo.get(sympy_object, default)

    def getSympySymbol(self, pyomo_object):
        if pyomo_object in self.pyomo2sympy:
            return self.pyomo2sympy[pyomo_object]
        # Pyomo currently ONLY supports Real variables (not complex
        # variables).  If that ever changes, then we will need to
        # revisit hard-coding the symbol type here
        sympy_obj = sympy.Symbol("x%d" % self.i, real=True)
        self.i += 1
        self.pyomo2sympy[pyomo_object] = sympy_obj
        self.sympy2pyomo[sympy_obj] = pyomo_object
        return sympy_obj

    def sympyVars(self):
        return self.sympy2pyomo.keys()


# =====================================================
# sympyify_expression
# =====================================================


class Pyomo2SympyVisitor(EXPR.StreamBasedExpressionVisitor):
    def __init__(self, object_map, keep_mutable_parameters=False):
        sympy.Add  # this ensures _configure_sympy gets run
        super(Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map
        self.keep_mutable_parameters = keep_mutable_parameters

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)

    def exitNode(self, node, values):
        if node.__class__ is EXPR.UnaryFunctionExpression:
            return _functionMap[node._name](values[0])
        _op = _pyomo_operator_map.get(node.__class__, None)
        if _op is None:
            return node._apply_operation(values)
        else:
            return _op(*tuple(values))

    def beforeChild(self, node, child, child_idx):
        #
        # Don't replace native or sympy types
        #
        if type(child) in native_types:
            return False, child
        #
        # Replace pyomo variables with sympy variables
        #
        if child.is_potentially_variable():
            #
            # We will descend into all expressions...
            #
            if child.is_expression_type():
                return True, None
            else:
                return False, self.object_map.getSympySymbol(child)
        #
        # Everything else is a constant...
        #
        if self.keep_mutable_parameters and child.is_parameter_type() and child.mutable:
            return False, self.object_map.getSympySymbol(child)
        return False, value(child)


class Sympy2PyomoVisitor(EXPR.StreamBasedExpressionVisitor):
    def __init__(self, object_map):
        sympy.Add  # this ensures _configure_sympy gets run
        super(Sympy2PyomoVisitor, self).__init__()
        self.object_map = object_map

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)

    def enterNode(self, node):
        return (node.args, [])

    def exitNode(self, node, values):
        """Visit nodes that have been expanded"""
        _op = _operatorMap.get(node.func, None)
        if _op is None:
            raise DeveloperError(
                f"sympy expression type {node.func} not found in the operator map"
            )
        return _op(tuple(values))

    def beforeChild(self, node, child, child_idx):
        if not child.args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return False, item
        return True, None


def sympyify_expression(expr, keep_mutable_parameters=False):
    """Convert a Pyomo expression to a Sympy expression"""
    #
    # Create the visitor and call it.
    #
    object_map = PyomoSympyBimap()
    visitor = Pyomo2SympyVisitor(
        object_map, keep_mutable_parameters=keep_mutable_parameters
    )
    return object_map, visitor.walk_expression(expr)


def sympy2pyomo_expression(expr, object_map):
    visitor = Sympy2PyomoVisitor(object_map)
    return visitor.walk_expression(expr)

# -*- coding: utf-8 -*-
"""sympy_wolfram: read Wolfram Mathematica, write SymPy.

Generic Wolfram support. It knows nothing about integration, about Rubi, or about
any other domain -- callers supply that context (see ``reserved_symbols`` and
``custom_functions``).

The package is split by ROLE, in pipeline order:

``parser``
    Mathematica source text -> Full-Form List (FFL). Pure syntax: every node is a
    plain string head with plain arguments, and no meaning is assigned.

``interpreter``
    FFL -> SymPy code strings and objects. This is where meaning is assigned:
    which Wolfram head maps to which SymPy function, which names are pattern
    wildcards, and what the evaluation namespace contains.

``objects``
    The Mathematica objects the interpreter can emit -- ``MathematicaExpr`` and
    its subclasses (``With``, ``Module``, ``Set``, ``Condition``, ...): the
    Wolfram constructs that have no direct SymPy equivalent and are modelled here.
"""

from .parser import mathematica_to_ffl
from .interpreter import (
    FFLConverter,
    ffl_to_sympy_code,
    ffl_to_sympy_short_code,
    mathematica_to_sympy,
    mathematica_to_sympy_code,
    mathematica_to_sympy_short_code,
)
from .objects import (
    Block,
    Catch,
    CompoundExpression,
    Do,
    Gamma,
    Head,
    If,
    List,
    MathematicaExpr,
    Module,
    Null,
    Reap,
    Return,
    Scan,
    Set,
    SetDelayed,
    Sow,
    Throw,
    With,
)
from .constraints import MathematicaConstraint
from .mathematica_functions import (
    Apply,
    Binomial,
    Coefficient,
    Complex,
    EllipticPi,
    Floor,
    FullSimplify,
    FunctionExpand,
    GCD,
    Hypergeometric2F1,
    LeafCount,
    Length,
    Not,
    Numerator,
    PolynomialQuotient,
    PolynomialRemainder,
    ProductLog,
    Quotient,
    ReplaceAll,
    Rule,
    Sign,
    Simplify,
    Sum,
    SumWolfram,
    Together,
)

__all__ = [
    # parser: text -> FFL
    'mathematica_to_ffl',
    # interpreter: FFL -> SymPy
    'FFLConverter',
    'ffl_to_sympy_code', 'ffl_to_sympy_short_code',
    'mathematica_to_sympy', 'mathematica_to_sympy_code',
    'mathematica_to_sympy_short_code',
    # objects: the modelled Mathematica constructs
    'Block', 'Catch', 'CompoundExpression', 'Do', 'Gamma', 'Head', 'If', 'List',
    'MathematicaExpr', 'Module', 'Null', 'Reap', 'Return', 'Scan', 'Set', 'SetDelayed', 'Sow',
    'Throw', 'With',
    # constraints: the Wolfram-predicate base class
    'MathematicaConstraint',
    # standard Wolfram function nodes
    'Apply', 'Binomial', 'Coefficient', 'Complex', 'EllipticPi', 'Floor',
    'FullSimplify', 'FunctionExpand', 'GCD', 'Hypergeometric2F1', 'LeafCount',
    'Length', 'Not', 'Numerator', 'PolynomialQuotient', 'PolynomialRemainder',
    'ProductLog', 'Quotient', 'ReplaceAll', 'Rule', 'Sign', 'Simplify', 'Sum',
    'SumWolfram', 'Together',
]

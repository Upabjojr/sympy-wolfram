# -*- coding: utf-8 -*-
"""Wolfram-flavoured base class for pattern constraints.

The GENERIC constraint base -- :class:`~sympy_matching.constraint.SymPyMatchingConstraint`
-- and its helpers (``_normalize_constraint_arg``, ``_collect_wildcards_from_args``,
``_resolve_with_substitution``) live in ``sympy_matching`` now, since they are not
Wolfram-specific: they are the constraint half of the reusable "SymPy + WildSymbol ->
omnimatch ManyToOneReplacer" machinery.

``MathematicaConstraint`` here is a THIN Wolfram-layer subclass that merely adds the
:class:`~sympy_wolfram.objects.MathematicaExpr` node identity on top of
``SymPyMatchingConstraint`` (so a Wolfram predicate is a first-class Mathematica-inspired
SymPy node, like every other object in this package). All the machinery -- argument
normalisation, ``variables``, ``check`` protocol, ``_resolve`` -- is inherited unchanged
from ``SymPyMatchingConstraint``.

The helpers are re-exported from this module for backward compatibility (several
``rubi_integrate`` modules and tests import ``_resolve_with_substitution`` etc. from here).
"""
from typing import Tuple

import sympy
from sympy.logic.boolalg import Boolean

from sympy_wolfram.objects import MathematicaExpr

# Generic base + helpers now live in sympy_matching; re-exported here so existing
# `from sympy_wolfram.constraints import _resolve_with_substitution` imports keep working.
from sympy_matching.constraint import (
    SymPyMatchingConstraint,
    _normalize_constraint_arg,
    _collect_wildcards_from_args,
    _resolve_with_substitution,
)

__all__ = [
    'MathematicaConstraint',
    'SymPyMatchingConstraint',
    '_normalize_constraint_arg',
    '_collect_wildcards_from_args',
    '_resolve_with_substitution',
]


class MathematicaConstraint(MathematicaExpr, SymPyMatchingConstraint):
    """A Wolfram-language pattern constraint.

    Adds the :class:`~sympy_wolfram.objects.MathematicaExpr` identity on top of
    :class:`~sympy_matching.constraint.SymPyMatchingConstraint`; everything else
    (arg normalisation, ``variables``, ``check``, ``_resolve*``) is inherited.

    MRO is ``MathematicaConstraint -> MathematicaExpr -> Expr -> ... ->
    SymPyMatchingConstraint -> Boolean -> Basic``. Because ``Expr`` precedes
    ``SymPyMatchingConstraint`` there, the four members ``Expr`` would otherwise
    supply -- ``__new__``, ``doit``, ``_evaluate``, ``free_symbols`` -- are
    re-declared here so the constraint (not the generic Expr) behaviour wins;
    ``variables``/``check``/``_resolve*`` are unique to ``SymPyMatchingConstraint``
    and resolve to it naturally.
    """

    # MathematicaExpr grants a __dict__ (declares no __slots__), so subclass state set
    # in __init__ (_var_name, _value, ...) works without re-declaring __slots__.

    def __new__(cls, *args, **kwargs):
        # Use the generic constraint constructor (normalise args, build via Boolean),
        # NOT Expr.__new__ which Expr would otherwise supply first in the MRO.
        return SymPyMatchingConstraint.__new__(cls, *args, **kwargs)

    def doit(self, **kwargs):
        # A constraint is a predicate, not a reducible expression; MathematicaExpr's
        # doit would deep-evaluate the args. Keep the node intact -- truth comes from check().
        return self

    def _evaluate(self, **kwargs):
        return self

    # SymPyMatchingConstraint.free_symbols returns set(), but Expr.free_symbols precedes
    # it in the MRO, so re-declare it here.
    @property
    def free_symbols(self):
        return set()

    # Structural constraints (MatchQ and friends) must NOT evaluate their pattern
    # argument -- they inspect the unevaluated tree. They set this to False.
    _EVAL_RESOLVED_ARGS = True

    def _resolve(self, expr, substitution):
        """Resolve *and evaluate* a constraint argument.

        The generated rules pass constraint arguments that are DEFERRED nodes built
        from the rule text -- ``GtQ(Expon(Pq_, x), 1)``, ``FreeQ(D(u_, x), x)``,
        ``SumQ(ExpandIntegrand(u_, x))`` … The generic resolution only substitutes the
        matched wildcards; without a ``doit()`` the predicate then receives an
        UNEVALUATED node and silently returns the wrong constant (an unevaluated
        relational is never provably true; ``D(3*x, x)`` still "contains" x; an
        ``ExpandIntegrand`` node is never an Add). That disabled or mis-fired several
        hundred rule guards across the ruleset -- in BOTH polarities, since
        ``Not(...)`` turns a vacuous False into a vacuous True.

        In Mathematica arguments are always evaluated before the predicate applies,
        so evaluating here is the faithful semantics. Only Wolfram-layer nodes are
        affected (guarded by ``has(MathematicaExpr)``); plain SymPy values pass
        through untouched, and structural constraints opt out via
        ``_EVAL_RESOLVED_ARGS = False``.
        """
        resolved = _resolve_with_substitution(expr, substitution)
        if self._EVAL_RESOLVED_ARGS and isinstance(resolved, sympy.Basic):
            try:
                if resolved.has(MathematicaExpr):
                    evaluated = resolved.doit()
                    # doit() of some nodes returns a PLAIN Python number (int/float);
                    # downstream predicates expect SymPy (eager_SumQ does .is_Add).
                    # Re-sympify numerics; leave lists/bools (SplitProduct, TrueQ)
                    # as-is -- the predicates handle those natively.
                    if not isinstance(evaluated, sympy.Basic) and isinstance(evaluated, (int, float, complex)):
                        evaluated = sympy.sympify(evaluated)
                    # A BooleanAtom NESTED inside arithmetic marks a FAILED evaluation:
                    # several Rubi helpers signal "no result" by returning False, and if
                    # such a node sits under e.g. a negation, doit(deep=True) rebuilds
                    # the parent as Mul(-1, False) -- which sympy 1.x still constructs
                    # (with a deprecation warning) and which then drives simplify into
                    # infinite recursion, aborting the whole integration with a
                    # RecursionError (seen on Int[(c+d x)^4 Gamma[n, a+b x]]). Keep the
                    # UNEVALUATED form instead: the guard then compares symbolically and
                    # comes out False, which is what Mathematica does. A boolean as the
                    # WHOLE result stays legitimate -- predicates handle those natively.
                    from sympy.logic.boolalg import BooleanAtom
                    poisoned = (isinstance(evaluated, sympy.Basic)
                                and not isinstance(evaluated, BooleanAtom)
                                and evaluated.atoms(BooleanAtom))
                    if not poisoned:
                        resolved = evaluated
            except Exception:
                pass
        return resolved

    @property
    def variables(self) -> Tuple[str, ...]:
        return tuple(_collect_wildcards_from_args(self.args))

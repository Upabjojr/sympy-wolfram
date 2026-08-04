# -*- coding: utf-8 -*-
"""Mathematica pattern semantics: ``d`` vs ``d_`` vs ``d_.`` must stay distinct.

Rubi's rules lean on three DIFFERENT things that all print as the letter ``d``:

===============  ==========================  ======================================
Mathematica      meaning                     our translation
===============  ==========================  ======================================
``d``            a LITERAL symbol            ``sympy.Symbol('d')``
``d_``           a plain Blank -- must be     ``WildSymbol('d_')``
                 present, binds ``d``
``d_.``          an Optional Blank -- MAY be  ``WildSymbol('d_', optional_value=
                 absent, then takes the       IDENTITY_ELEMENT)``
                 enclosing operation's
                 identity
===============  ==========================  ======================================

Collapsing any two of them changes which integrands a rule fires on, and a rule that
fires where Rubi's would not is exactly how a wrong antiderivative gets produced. The
subtlety that makes this worth pinning down: within ONE pattern the same NAME can be
both a literal and a pattern variable, and Mathematica keeps them INDEPENDENT --
``MatchQ[d + 5 W, d + d_.*W]`` is True, binding the pattern variable to 5 while the
literal ``d`` matched itself.

EVERY expected value below was read off Mathematica 12.2 (see the table in each test),
never derived from this port's behaviour.
"""
import pytest
from sympy import Symbol, sin, sqrt

from omnimatch import ManyToOneMatcher, Pattern

# A registered binary wrapper head so the rule machinery runs. This used to be
# rubi_integrate.base_objects.Int, but sympy-wolfram's tests must not depend on the
# higher-level integration package -- the three lines below are its definition.
import sympy as _sympy
from omnimatch.expressions.expressions import OperationHead as _OperationHead, Arity as _Arity
from sympy_matching.conversion import register_sympy_head as _register_sympy_head


class Int(_sympy.Function):
    nargs = 2


_register_sympy_head(Int, _OperationHead(name='Int', arity=_Arity.binary))
from sympy_matching.matching_rule import to_omnimatch_expression
from sympy_matching.wild import IDENTITY_ELEMENT, WildSymbol

x = Symbol('x')
W = Symbol('W')
d = Symbol('d')                                        # a LITERAL symbol
y = Symbol('y')

a_ = WildSymbol('a_')                                  # Mathematica  a_
_a_ = WildSymbol('a_', optional_value=IDENTITY_ELEMENT)   # Mathematica  a_.
d_ = WildSymbol('d_')                                  # Mathematica  d_
_d_ = WildSymbol('d_', optional_value=IDENTITY_ELEMENT)   # Mathematica  d_.
c_ = WildSymbol('c_')


def matches(pattern, subject):
    """True if `pattern` matches `subject`, wrapped in Int so the rule machinery runs."""
    matcher = ManyToOneMatcher()
    matcher.add(Pattern(to_omnimatch_expression(Int(pattern, x))))
    return bool(list(matcher.match(to_omnimatch_expression(Int(subject, x)))))


# ── 1. an Optional Blank may be ABSENT, defaulting to the operation's identity ──
# Mathematica: MatchQ[W, a_. + W]  True ({a, 0})
#              MatchQ[W, a_.*W]    True ({a, 1})
#              MatchQ[W, W^a_.]    True ({a, 1})
@pytest.mark.parametrize('label, pattern', [
    ('Plus  -> identity 0', _a_ + W),
    ('Times -> identity 1', _a_*W),
    ('Power -> identity 1', W**_a_),
])
def test_optional_blank_may_be_absent(label, pattern):
    assert matches(pattern, W) is True, label


# ── 2. a plain Blank may NOT be absent ────────────────────────────────────────
# Mathematica: MatchQ[W, a_ + W]  False;  MatchQ[W, a_*W]  False
@pytest.mark.parametrize('label, pattern', [
    ('Plus', a_ + W),
    ('Times', a_*W),
])
def test_plain_blank_may_not_be_absent(label, pattern):
    assert matches(pattern, W) is False, label


# ── 3. the same NAME as Blank AND Optional must bind CONSISTENTLY ─────────────
# This is Rubi's `d_ + d_.*ProductLog[...]` shape -- the denominator is d*(1 + W), so
# both occurrences must agree. Mathematica:
#   5 + 5 W -> True     2 + 3 W -> False    1 + W -> True ({d, 1})
#   5 + W   -> False    W       -> False
@pytest.mark.parametrize('label, subject, expected', [
    ('d=5 in both',                 5 + 5*W, True),
    ('2 vs 3 -- inconsistent',      2 + 3*W, False),
    ('d=1, optional term absent',   1 + W,   True),
    ('5 vs implied 1 -- clashes',   5 + W,   False),
    ('plain Blank cannot vanish',   W,       False),
])
def test_same_name_blank_and_optional_must_agree(label, subject, expected):
    assert matches(d_ + _d_*W, subject) is expected, label


# ── 4. a LITERAL symbol and a pattern variable of the same name are INDEPENDENT ─
# Mathematica: MatchQ[d + d W, d + d_.*W]  True
#              MatchQ[5 + 5 W, d + d_.*W]  False   (literal d does not match 5)
#              MatchQ[d + 5 W, d + d_.*W]  True    (!! d_. binds 5, literal d matched d)
@pytest.mark.parametrize('label, subject, expected', [
    ('literal d, variable binds d', d + d*W, True),
    ('literal d cannot match 5',    5 + 5*W, False),
    ('variable binds 5 while the literal matches itself', d + 5*W, True),
])
def test_literal_symbol_is_independent_of_a_same_named_wildcard(label, subject, expected):
    assert matches(d + _d_*W, subject) is expected, label


# ── 4b. `d_` and `d_.` are the SAME VARIABLE; optionality belongs to the SLOT ──
# Mathematica FullForm makes this explicit:
#     d_   ->  Pattern[d, Blank[]]
#     d_.  ->  Optional[Pattern[d, Blank[]]]        <- Optional WRAPS the same Pattern
# so there is ONE binding (`ReplaceList[5+5W, d_ + d_.*W :> d]` returns `{5}`), and when
# a slot is absent its DEFAULT has to agree with whatever the other slots bound.
# Mathematica:
#     MatchQ[W,     d_. + d_.*W]  False   (Plus default 0 vs Times default 1 -- clash)
#     MatchQ[3 W,   d_. + d_.*W]  False   (0 vs 3)
#     MatchQ[5+5 W, d_. + d_.*W]  True    (5 and 5)
@pytest.mark.parametrize('label, subject, expected', [
    ('both slots default: 0 vs 1 clash', W,       False),
    ('Plus default 0 vs Times 3',        3*W,     False),
    ('both bind 5',                      5 + 5*W, True),
])
def test_one_optional_variable_used_twice_must_agree_including_defaults(label, subject, expected):
    assert matches(_d_ + _d_*W, subject) is expected, label


# Mathematica: MatchQ[5+5W, d_ + d_*W] True;  MatchQ[2+3W, d_ + d_*W] False
@pytest.mark.parametrize('label, subject, expected', [
    ('both bind 5', 5 + 5*W, True),
    ('2 vs 3',      2 + 3*W, False),
])
def test_one_plain_variable_used_twice_must_agree(label, subject, expected):
    assert matches(d_ + d_*W, subject) is expected, label


# ── 4c. the shared NAME is what unifies the two slots -- in EVERY shape ───────
# `d_` and `d_.` are one variable, and our translation gives them two DISTINCT SymPy
# objects that share a omnimatch variable NAME. These check that the unification is a
# property of the name and not an accident of one flat `Add`, so no explicit
# `Eq(d_, _d_)` constraint is needed to hold the two slots together.
@pytest.mark.parametrize('label, pattern, subject, expected', [
    ('flat Add',        d_ + _d_*W,          5 + 5*W,          True),
    ('flat Add clash',  d_ + _d_*W,          2 + 3*W,          False),
    ('nested in sin',   sin(d_) + _d_*W,     sin(5) + 5*W,     True),
    ('nested clash',    sin(d_) + _d_*W,     sin(2) + 3*W,     False),
    ('under sqrt',      sqrt(d_) + _d_*W,    sqrt(5) + 5*W,    True),
    ('two Muls',        d_*y + _d_*W,        5*y + 5*W,        True),
    ('two Muls clash',  d_*y + _d_*W,        2*y + 3*W,        False),
    ('deep nesting',    sin(d_*y) + _d_*W,   sin(5*y) + 5*W,   True),
])
def test_the_shared_name_unifies_the_slots_in_every_shape(label, pattern, subject, expected):
    assert matches(pattern, subject) is expected, label


# Mathematica does NOT solve equations while matching: `MatchQ[25, d_^2]` is False,
# because 25 is an Integer and not a Power. A SYMBOLIC square does match.
@pytest.mark.parametrize('label, subject, expected', [
    ('25 is not a Power -- no equation solving', 25 + 5*W,      False),
    ('symbolic square matches',                  y**2 + y*W,    True),
    ('4 is not a Power either',                  4 + 3*W,       False),
])
def test_matching_never_solves_for_a_wildcard(label, subject, expected):
    assert matches(d_**2 + _d_*W, subject) is expected, label


# ── 5. two DIFFERENT names bind independently ────────────────────────────────
# Mathematica: MatchQ[2 + 3 W, c_ + d_.*W]  True
def test_distinct_names_bind_independently():
    assert matches(c_ + _d_*W, 2 + 3*W) is True


# ── 6. the shape that motivated all of this: Rubi 8.9 rule 40 ────────────────
def test_rubi_8_9_rule_40_denominator_requires_one_d():
    """`Int[x^m (c W[a x^n])^p / (d + d_. W[a x^n])]` -- the denominator is d(1+W).

    A denominator whose two coefficients differ is NOT this rule's shape; firing on it
    would apply a replacement derived for d(1+W) to something else entirely.
    """
    from sympy import LambertW
    import rubi_integrate.rules.r_8_special_functions.r_8_9 as mod
    rule40 = next(r for r in mod.RULES if r.rule_number == 40)
    matcher = ManyToOneMatcher()
    matcher.add(Pattern(to_omnimatch_expression(rule40.pattern)))

    a, c = Symbol('a'), Symbol('c')
    lam = LambertW(a*x**2)

    def fires(integrand):
        return bool(list(matcher.match(to_omnimatch_expression(Int(integrand, x)))))

    assert fires(x**3*lam**2/(5 + 5*lam)) is True     # d = 5
    assert fires(x**3*lam**2/(1 + lam)) is True       # d = 1, optional absent
    assert fires(x**3*lam**2/(2 + 3*lam)) is False    # coefficients disagree
    assert fires(x**3*lam**2/(7 + lam)) is False      # 7 vs implied 1


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATOR-level tests: `ffl_to_sympy_code` must itself distinguish the three
# forms. The tests above check that hand-built WildSymbols behave like Mathematica;
# these check that the TRANSLATOR produces the right ones from Wolfram FullForm --
# which is what any other consumer of `sympy_wolfram` depends on.
# ═══════════════════════════════════════════════════════════════════════════════

from sympy_wolfram.interpreter import ffl_to_sympy_code   # noqa: E402


def _blank(name):
    return ['Pattern', name, ['Blank']]


def _optional(name):
    return ['Optional', ['Pattern', name, ['Blank']]]


def test_translator_renders_the_three_forms_distinctly():
    """``d_`` -> ``d_``, ``d_.`` -> ``_d_``, bare ``d`` -> ``Symbol('d')``."""
    code, defs, _ = ffl_to_sympy_code(['Plus', _blank('d'), ['Times', _optional('d'), 'W']])
    assert code == "(d_ + (_d_ * Symbol('W')))"
    assert "d_ = WildSymbol('d')" in defs
    assert "_d_ = WildSymbol('d', optional_value=IDENTITY_ELEMENT)" in defs


@pytest.mark.parametrize('label, ffl, expected', [
    ('literal first',
     ['Plus', 'd', ['Times', ['Optional', ['Pattern', 'd', ['Blank']]], 'W']],
     "(Symbol('d') + (_d_ * Symbol('W')))"),
    ('literal last',
     ['Plus', ['Times', ['Optional', ['Pattern', 'd', ['Blank']]], 'W'], 'd'],
     "((_d_ * Symbol('W')) + Symbol('d'))"),
])
def test_a_literal_symbol_in_a_pattern_stays_literal_whatever_the_order(label, ffl, expected):
    """`Plus` is orderless, so these are the SAME Mathematica expression.

    The bare ``d`` is a LITERAL symbol in both, independent of the same-named pattern
    variable (Mathematica: ``MatchQ[d + 5 W, d + d_.*W]`` is True). This used to be
    order-dependent: the sets of discovered wildcards fill up as the walk proceeds, so a
    literal appearing AFTER the wildcard was rewritten INTO it, yielding
    ``_d_*W + _d_`` -- which demands both be equal and no longer matches ``d + 5 W``.
    """
    code, _, _ = ffl_to_sympy_code(ffl)
    assert code == expected, label


@pytest.mark.parametrize('kwargs, expected', [
    ({'wildcards': {'d'}}, "(d_ * Symbol('W'))"),
    ({'optional_wildcards': {'d'}}, "(_d_ * Symbol('W'))"),
])
def test_bare_atoms_still_resolve_to_wildcards_on_the_replacement_side(kwargs, expected):
    """A replacement/constraint FFL has no ``Pattern[...]`` nodes -- its wildcards are
    bare atoms and MUST resolve to the bound values. The caller signals this by
    pre-seeding the names, and that path must keep working."""
    code, _, _ = ffl_to_sympy_code(['Times', 'd', 'W'], **kwargs)
    assert code == expected


def test_translated_pattern_matches_exactly_what_mathematica_matches():
    """End-to-end: translate the Wolfram pattern, then match with it.

    Mathematica 12.2 values for ``d + d_.*W``:
        d + d W -> True,  5 + 5 W -> False,  d + 5 W -> True
    """
    ns = {}
    code, _, _ = ffl_to_sympy_code(
        ['Plus', 'd', ['Times', ['Optional', ['Pattern', 'd', ['Blank']]], 'W']],
        namespace=ns)
    pattern = eval(code, ns)                                    # noqa: S307 - trusted codegen
    assert matches(pattern, d + d*W) is True
    assert matches(pattern, 5 + 5*W) is False
    assert matches(pattern, d + 5*W) is True

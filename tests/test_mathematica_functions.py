# -*- coding: utf-8 -*-
"""Tests for the standard Wolfram function nodes moved into sympy_wolfram.

These functions (GCD, Sign, Floor, Together, ProductLog, LeafCount, …) are
Mathematica standard-library, not Rubi-specific, so they live in
``sympy_wolfram.mathematica_functions`` (deferred nodes) and
``sympy_wolfram.functions_eager`` (self-contained eager helpers).

Covers:
- Module location: the nodes live in sympy_wolfram, not rubi_integrate
- No wrong-direction import: neither new module imports rubi_integrate
- Behaviour of a representative set (via .doit())
- Backward-compat: rubi_integrate.utils.rubi_utils re-exports the SAME class objects
"""
import ast
import importlib
import math

import pytest
import sympy
from sympy import Symbol, Integer, I, S, sin, cos

from sympy_wolfram import mathematica_functions as mf
from sympy_wolfram import functions_eager as fe
from sympy_wolfram.objects import MathematicaExpr


x = Symbol('x')


# ---------------------------------------------------------------------------
# 1. Location + no wrong-direction imports
# ---------------------------------------------------------------------------

MOVED_NODES = [
    'Coefficient', 'PolynomialQuotient', 'PolynomialRemainder', 'Rule',
    'ReplaceAll', 'SumWolfram', 'Numerator', 'Together', 'GCD', 'Sign',
    'Quotient', 'EllipticPi', 'Apply', 'FullSimplify', 'Simplify',
    'FunctionExpand', 'Binomial', 'ProductLog', 'Floor', 'Hypergeometric2F1',
    'LeafCount', 'Length', 'Not',
    # Relocated out of rubi_integrate (their Rubi bodies only ever called generic
    # SymPy operations — is_Add/is_Mul/sort_key/is_rational_function/part-extract).
    'First', 'Rest', 'Part', 'Apart', 'Denominator', 'Exponent',
]


@pytest.mark.parametrize('name', MOVED_NODES)
def test_node_lives_in_sympy_wolfram(name):
    """EVERY Wolfram standard-library head here is a deferred MathematicaExpr.

    sympy_wolfram is an interpreter for the Wolfram language and the runtime library
    translated code links against, so a head keeps its Wolfram identity and evaluates
    only on doit() -- even when SymPy happens to provide the same function.
    """
    cls = getattr(mf, name)
    assert cls.__module__ == 'sympy_wolfram.mathematica_functions'
    assert issubclass(cls, MathematicaExpr)


@pytest.mark.parametrize('mod', [
    'sympy_wolfram.mathematica_functions',
    'sympy_wolfram.functions_eager',
])
def test_no_rubi_import_in_module(mod):
    m = importlib.import_module(mod)
    tree = ast.parse(open(m.__file__).read())
    rubi = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and any('rubi' in getattr(a, 'name', '') or 'rubi' in (getattr(n, 'module', '') or '')
                for a in getattr(n, 'names', [n]))
    ]
    assert not rubi, f"{mod} imports rubi_integrate: {rubi}"


# ---------------------------------------------------------------------------
# 2. Behaviour
# ---------------------------------------------------------------------------

def test_behaviour():
    assert mf.GCD(Integer(12), Integer(18)).doit() == Integer(6)
    assert mf.Sign(Integer(-3)).doit() == Integer(-1)
    assert mf.Floor(Integer(7), Integer(2)).doit() == Integer(6)     # nearest multiple of 2
    assert mf.Quotient(Integer(7), Integer(2)).doit() == Integer(3)
    assert mf.ProductLog(Integer(0)).doit() == Integer(0)
    assert mf.Binomial(Integer(5), Integer(2)).doit() == Integer(10)
    assert mf.Coefficient(x**2 + 3 * x, x, Integer(1)).doit() == Integer(3)
    assert mf.Sum(x, mf.List(x, Integer(1), Integer(3))).doit() == Integer(6)
    assert mf.LeafCount(sin(x)).doit() == Integer(2)
    assert mf.Length(x + Integer(1)).doit() == Integer(2)
    assert mf.Not(Integer(0)).doit() is True
    assert mf.Complex(Integer(2), Integer(3)) == 2 + 3 * I           # eager __new__


def test_SumWolfram_floors_fractional_bounds():
    """Regression: the binomial-Pq rules build Sum(coeff, {k, 0, (q-r)/n}), whose upper
    limit is FRACTIONAL when the degree doesn't divide evenly (e.g. (8-3)/4 = 5/4).
    SumWolfram must truncate the iterator at floor(imax) (Mathematica semantics) and
    EXPAND to finite terms -- a fractional-bound sympy.Sum stays UNEVALUATED and drove
    simplify() into unbounded recursion, crashing Int[(x^4+1)/(x^8+1)] and
    Int[(d+e x^4)/(a-c x^8)] with RecursionError."""
    from sympy import Rational
    k = sympy.Symbol('k')
    # fractional upper bound 5/4 -> iterate k = 0, 1 (floor)
    assert mf.Sum(x**(4 * k) * x**(4 * k + 3), mf.List(k, Integer(0), Rational(5, 4))).doit() == x**11 + x**3
    # integer bound unchanged
    assert mf.Sum(x**k, mf.List(k, Integer(0), Integer(3))).doit() == x**3 + x**2 + x + 1
    # a SYMBOLIC bound is only floored when concrete: it must NOT be truncated and
    # still evaluate normally (sympy's geometric closed form), depending on q.
    q = sympy.Symbol('q')
    assert mf.Sum(x**k, mf.List(k, Integer(0), q / 2)).doit().has(q)


@pytest.mark.parametrize('expr', [sin(x), cos(x), sympy.exp(x), sympy.log(x), sympy.tan(x)])
def test_eager_PolynomialQ_is_total_over_transcendentals(expr):
    """Mathematica's PolynomialQ is TOTAL: PolynomialQ[Sin[x], x] is False, not unknown.

    SymPy's ``is_polynomial`` answers None ("undecided") for every transcendental
    function of x, and that None used to leak straight out. It is falsy, so a plain
    guard behaved correctly by accident -- but ``Not[PolynomialQ[Sin[x], x]]`` then
    came back None instead of True (eager_Not propagates None), and None is falsy,
    so a NEGATED guard that Rubi passes we failed, silently disabling those rules.
    """
    assert fe.eager_PolynomialQ(expr, x) is False
    assert fe.eager_Not(fe.eager_PolynomialQ(expr, x)) is True


def test_eager_PolynomialQ_still_decides_the_easy_cases():
    assert fe.eager_PolynomialQ(x**2 + 1, x) is True
    assert fe.eager_PolynomialQ(1/x, x) is False
    assert fe.eager_PolynomialQ(sympy.sqrt(x), x) is False
    assert fe.eager_PolynomialQ(sin(a), x) is True      # free of x -> a constant


def test_eager_helpers_are_self_contained():
    assert fe.eager_LeafCount(sin(x)) == 2
    assert fe.eager_Length(x + Integer(1)) == 2
    assert fe.eager_Complex(Integer(0), Integer(1)) == I
    assert fe.eager_Not(False) is True


# ---------------------------------------------------------------------------
# 2b. Relocated First/Rest/Part/Apart/Numerator/Denominator/Exponent/Simplify
#     (behaviour moved here from rubi_integrate/tests/test_utility_function.py, since
#     the functions themselves moved into this layer).
# ---------------------------------------------------------------------------

a, b, c, y = sympy.symbols('a b c y')


def test_eager_First_Rest():
    assert fe.eager_First([2, 3, 5, 7]) == 2
    assert fe.eager_First(y ** 2) == y
    assert fe.eager_First(a + b + c) == a          # canonical sort_key order
    assert fe.eager_First(a * b * c) == a
    assert fe.eager_Rest([2, 3, 5, 7]) == [3, 5, 7]
    assert fe.eager_Rest(a + b + c) == b + c
    assert fe.eager_Rest(a * b * c) == b * c
    assert fe.eager_Rest(1 / b) == -1


def test_eager_Numerator_Denominator():
    assert fe.eager_Numerator((-a / b) ** 3) == (-a) ** 3
    assert fe.eager_Numerator(S(3) / 2) == 3
    assert fe.eager_Numerator(x / y) == x
    assert fe.eager_Numerator(-S(1) / 2 + I / 3) == -3 + 2 * I
    assert fe.eager_Denominator((-a / b) ** 3) == b ** 3
    assert fe.eager_Denominator(S(3) / 2) == 2
    assert fe.eager_Denominator(x / y) == y
    assert fe.eager_Denominator(-S(1) / 2 + I / 3) == 6


def test_eager_Part():
    assert fe.eager_Part([1, 2, 3], 1) == 1
    assert fe.eager_Part(a * b, 1) == a
    assert fe.Util_Part(1, a + b).doit() == a
    assert fe.Util_Part(c, a + b).doit() == fe.Util_Part(c, a + b)   # symbolic index -> deferred


def test_eager_Apart():
    assert fe.eager_Apart(1 / (x ** 2 * (a + b * x) ** 2), x) == (
        b ** 2 / (a ** 2 * (a + b * x) ** 2) + 1 / (a ** 2 * x ** 2)
        + 2 * b ** 2 / (a ** 3 * (a + b * x)) - 2 * b / (a ** 3 * x))
    # Non-rational: returned unchanged (matches Mathematica, guards SymPy's apart).
    assert fe.eager_Apart(x ** (S(2) / 3) * (a + b * x) ** 2, x) == x ** (S(2) / 3) * (a + b * x) ** 2


def test_eager_Exponent_is_rational_function_faithful():
    assert fe.eager_Exponent(x ** 3 + x + 1, x) == 3
    assert fe.eager_Exponent(x ** 2 + 2 * x + 1, x) == 2
    assert fe.eager_Exponent(S(1), x) == 0
    # Mathematica treats the argument as a rational function: Exponent[x^-3, x] == -3
    # (a polynomial-only implementation would wrongly return 0).
    assert fe.eager_Exponent(x ** (-3), x) == -3


def test_eager_Exponent_power_form():
    """A power ``form`` counts by exponent RATIO, as in Mathematica:
    Exponent[x^6, x^2] == 3, Exponent[x^5, x^2] == 5/2, Exponent[x, Sqrt[x]] == 2.
    Counting only exact-base factors returned 1 for Exponent[5x^6+3x^4+x^2+4, x^2],
    which silently failed every ``Expon[..., x^2] > 1`` guard and disabled the
    even-polynomial rule family (1.2.2.5-7): plain rational integrands like
    x^2 (5x^6+3x^4+x^2+4)/(x^4+3x^2+2)^3 fell through to the Unintegrable catch-all."""
    poly = 5 * x ** 6 + 3 * x ** 4 + x ** 2 + 4
    assert fe.eager_Exponent(poly, x ** 2) == 3
    assert fe.eager_Exponent(x ** 2 * poly, x ** 2) == 4
    assert fe.eager_Exponent(x ** 5, x ** 2) == S(5) / 2
    assert fe.eager_Exponent(x, x ** 2) == S(1) / 2
    assert fe.eager_Exponent(x, sympy.sqrt(x)) == 2
    assert fe.eager_Exponent(S(4), x ** 2) == 0


def test_eager_Simplify():
    assert fe.eager_Simplify(sin(x) ** 2 + cos(x) ** 2) == 1
    assert fe.eager_Simplify((x ** 3 + x ** 2 - x - 1) / (x ** 2 + 2 * x + 1)) == x - 1


def test_eager_FreeQ():
    """FreeQ is a standard Wolfram predicate lifted here from rubi_integrate; a list is free
    iff every element is. It accepts either SymPy or match-bound OmniMatch values."""
    a, b, y = sympy.symbols('a b y')
    assert fe.eager_FreeQ(a + b * y, x) is True          # no x
    assert fe.eager_FreeQ(a + b * x, x) is False         # contains x
    assert fe.eager_FreeQ([a, b, y], x) is True          # all free
    assert fe.eager_FreeQ([a, b * x], x) is False        # one contains x
    # omnimatch SymbolWrapper coerces to its sympy value
    from omnimatch.expressions.expressions import SymbolWrapper
    assert fe.eager_FreeQ(SymbolWrapper(a), x) is True


def test_freeq_lifted_and_reexported():
    """FreeQ (and its omnimatch->sympy helper _ensure_sympy) live in sympy_wolfram now;
    rubi_integrate re-exports the SAME objects, and the Rubi FreeQ constraint delegates here."""
    import importlib
    uf = importlib.import_module('rubi_integrate.utils.utility_functions')
    assert uf.eager_FreeQ is fe.eager_FreeQ
    assert uf._ensure_sympy is fe._ensure_sympy


def test_eager_IntegerQ():
    """IntegerQ is a standard Wolfram predicate lifted here; True iff an explicit integer."""
    assert fe.eager_IntegerQ(S(1)) is True
    assert fe.eager_IntegerQ(S(-1)) is True
    assert fe.eager_IntegerQ(S(-1.9)) is False
    assert fe.eager_IntegerQ(S(0.0)) is False


def test_eager_AtomQ():
    """AtomQ is a standard Wolfram predicate lifted here; True iff no subexpressions."""
    assert fe.eager_AtomQ(x)
    assert not fe.eager_AtomQ(x + 1)
    assert not fe.eager_AtomQ([a, b])


def test_eager_NumberQ():
    """NumberQ is a standard Wolfram predicate lifted here: True ONLY for explicit numbers
    -- Integer/Rational/Real or Complex[a,b] with explicit parts. Cross-checked against
    real Rubi (ssh pi): Pi, E, Sqrt[2], (-1)^(1/4), Sqrt[2]*I are NOT numbers (symbolic
    constants / radicals), while I, 3*I and 2+3*I ARE. (SymPy's is_number is broader --
    it accepts every constant -- which used to make NumberQ[(-1)^(1/4)] wrongly True.)"""
    assert fe.eager_NumberQ(S(2))
    assert fe.eager_NumberQ(sympy.Rational(3, 2))
    assert fe.eager_NumberQ(sympy.sympify(2.5))
    assert fe.eager_NumberQ(I)
    assert fe.eager_NumberQ(3 * I)
    assert fe.eager_NumberQ(2 + 3 * I)
    assert not fe.eager_NumberQ(sympy.pi)
    assert not fe.eager_NumberQ(sympy.E)
    assert not fe.eager_NumberQ(sympy.sqrt(2))
    assert not fe.eager_NumberQ((-1) ** (S(1) / 4))
    assert not fe.eager_NumberQ(sympy.sqrt(2) * I)
    assert not fe.eager_NumberQ(-(-1) ** (S(3) / 4) + (-1) ** (S(1) / 4))  # really sqrt(2), but a Plus of Powers
    assert not fe.eager_NumberQ(x)
    assert not fe.eager_NumberQ(2 * x)


def test_eager_PolynomialQ():
    """PolynomialQ is a standard Wolfram predicate lifted here; polynomial test in a variable."""
    A, B, C = sympy.symbols('A B C')
    assert not fe.eager_PolynomialQ(x * (-1 + x ** 2), (1 + x) ** (S(1) / 2))
    assert not fe.eager_PolynomialQ((16 * x + 1) / ((x + 5) ** 2 * (x ** 2 + x + 1)), 2 * x)
    assert not fe.eager_PolynomialQ(A + b * x + c * x ** 2, x ** 2)
    assert fe.eager_PolynomialQ(A + B * x + C * x ** 2)
    assert fe.eager_PolynomialQ(A + B * x ** 4 + C * x ** 2, x ** 2)
    assert fe.eager_PolynomialQ(x ** 3, x)
    assert not fe.eager_PolynomialQ(sympy.sqrt(x), x)


def test_eager_PositiveQ():
    """PositiveQ is a standard Wolfram predicate lifted here; truthy iff a positive real.

    (A comparable value returns SymPy's ``BooleanTrue``/``BooleanFalse``, not a Python
    bool, so these use plain truthiness.)"""
    assert fe.eager_PositiveQ(S(1))
    assert not fe.eager_PositiveQ(S(-3))
    assert not fe.eager_PositiveQ(S(0))
    assert not fe.eager_PositiveQ(sympy.zoo)
    assert not fe.eager_PositiveQ(I)        # not comparable -> not positive
    d = sympy.Symbol('d')
    assert fe.eager_PositiveQ(b / (b * (b * c / (-a * d + b * c)) - a * (b * d / (-a * d + b * c))))


def test_eager_MemberQ():
    """MemberQ is a standard Wolfram predicate lifted here (plain membership)."""
    assert fe.eager_MemberQ([a, b, c], b) is True
    assert fe.eager_MemberQ([sin, cos, sympy.log, sympy.tan], sin(x).func) is True
    assert fe.eager_MemberQ([[sin, cos], [sympy.tan, sympy.cot]], [sin, cos]) is True
    assert fe.eager_MemberQ([[sin, cos], [sympy.tan, sympy.cot]], [sin, sympy.tan]) is False


def test_eager_MemberQ_head_wildcard_matches_by_class():
    """A function-head wildcard F_[...] binds its head to a HeadRef; MemberQ must fire
    against the head's class whether the list holds HeadRef literals (codegen form) or
    bare classes. Regression: these head-checks otherwise silently failed and the FHW
    rule never fired. (TrigQ/InverseTrigQ routing through this stays covered in the
    rubi_integrate utility-function tests.)"""
    from sympy_matching.wild import HeadRef
    from sympy import asin, acos, atan, erf, fresnels
    # codegen form: HeadRef literals in the list
    assert fe.eager_MemberQ([HeadRef(asin), HeadRef(acos)], HeadRef(asin)) is True
    assert fe.eager_MemberQ([HeadRef(asin), HeadRef(acos)], HeadRef(atan)) is False
    assert fe.eager_MemberQ([HeadRef(erf), HeadRef(fresnels)], HeadRef(fresnels)) is True
    # bare-class list, HeadRef subject
    assert fe.eager_MemberQ([sin, cos], HeadRef(sin)) is True


def test_predicates_reexported_by_rubi():
    """rubi_integrate re-exports the SAME eager predicate objects from this layer."""
    import importlib
    uf = importlib.import_module('rubi_integrate.utils.utility_functions')
    assert uf.eager_IntegerQ is fe.eager_IntegerQ
    assert uf.eager_PositiveQ is fe.eager_PositiveQ
    assert uf.eager_MemberQ is fe.eager_MemberQ


def test_head_to_class_unwraps_headref_and_class():
    """head_to_class is the structural bridge that lets a wildcard function head
    (bound as a HeadRef carrying its SymPy class) compare against a list of function
    classes. Mathematica->SymPy *name* translation happens in the code generator (it
    emits ``HeadRef(sympy.asin)``), so this only unwraps HeadRef / classes."""
    from sympy_matching.wild import HeadRef
    # HeadRef carries the class directly
    assert fe.head_to_class(HeadRef(sympy.asin)) is sympy.asin
    assert fe.head_to_class(HeadRef(sympy.fresnels)) is sympy.fresnels
    # a bare class round-trips
    assert fe.head_to_class(sympy.sin) is sympy.sin
    # not a head -> None
    assert fe.head_to_class(Symbol('x')) is None
    assert fe.head_to_class(sympy.Integer(3)) is None


def test_deferred_nodes_delegate_to_eager():
    assert mf.First(a + b + c).doit() == a
    assert mf.Rest(a * b * c).doit() == b * c
    assert mf.Part(sympy.Tuple(a, b, x), Integer(2)).doit() == b
    assert mf.Numerator((a + 1) / (b * x)).doit() == a + 1
    assert mf.Denominator((a + 1) / (b * x)).doit() == b * x
    assert mf.Apart(1 / (x * (x + 1)), x).doit() == 1 / x - 1 / (x + 1)
    assert mf.Exponent(a + b * x ** 3, x).doit() == 3


# ---------------------------------------------------------------------------
# 3. Backward-compat: rubi_utils re-exports the identical objects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', MOVED_NODES + ['Sum'])
def test_rubi_utils_reexports_same_object(name):
    ru = importlib.import_module('rubi_integrate.utils.rubi_utils')
    assert getattr(ru, name) is getattr(mf, name)


def test_gamma_deduplicated():
    """rubi_utils no longer defines its own Gamma; it uses sympy_wolfram's."""
    ru = importlib.import_module('rubi_integrate.utils.rubi_utils')
    assert ru.Gamma.__module__ == 'sympy_wolfram.objects'


# ---------------------------------------------------------------------------
# 4. Standard Wolfram special functions
#
# Every expected value below was produced by real Mathematica 12.2 and is quoted
# in the assertion comments. These are Wolfram-language BUILTINS (they appear
# nowhere in Rubi's IntegrationUtilityFunctions.m), so they belong in this layer
# rather than in rubi_integrate.
# ---------------------------------------------------------------------------

def test_eager_Factorial_matches_mathematica():
    from sympy_wolfram.functions_eager import eager_Factorial
    n = sympy.Symbol('n')
    assert eager_Factorial(5) == 120                       # Factorial[5] == 120
    assert eager_Factorial(0) == 1
    # MMA reduces a non-integer through Gamma: Factorial[1/2] == Sqrt[Pi]/2.
    # SymPy's factorial() alone leaves it unevaluated.
    assert eager_Factorial(sympy.Rational(1, 2)) == sympy.sqrt(sympy.pi) / 2
    # a symbolic argument stays symbolic (MMA prints n!)
    assert eager_Factorial(n) == sympy.factorial(n)


def test_eager_Zeta_matches_mathematica():
    from sympy_wolfram.functions_eager import eager_Zeta
    s, a = sympy.symbols('s a')
    assert eager_Zeta(2) == sympy.pi ** 2 / 6              # Zeta[2] == Pi^2/6
    assert eager_Zeta(s) == sympy.zeta(s)                  # Zeta[s] stays symbolic
    # Hurwitz form: Zeta[2, 3] == -5/4 + Pi^2/6
    assert sympy.simplify(eager_Zeta(2, 3) - (sympy.Rational(-5, 4) + sympy.pi ** 2 / 6)) == 0
    assert eager_Zeta(2, a) == sympy.zeta(2, a)


def test_eager_PolyGamma_matches_mathematica():
    from sympy_wolfram.functions_eager import eager_PolyGamma
    z = sympy.Symbol('z')
    # MMA's 1-arg form IS PolyGamma[0, z]
    assert eager_PolyGamma(z) == sympy.polygamma(0, z)
    assert eager_PolyGamma(0, z) == sympy.polygamma(0, z)
    assert eager_PolyGamma(2, z) == sympy.polygamma(2, z)
    assert eager_PolyGamma(1, 1) == sympy.pi ** 2 / 6      # PolyGamma[1,1] == Pi^2/6


def test_eager_BesselJ_matches_mathematica():
    from sympy_wolfram.functions_eager import eager_BesselJ
    n, z = sympy.symbols('n z')
    assert eager_BesselJ(1, z) == sympy.besselj(1, z)
    assert eager_BesselJ(n, z) == sympy.besselj(n, z)
    # MMA auto-expands half-integer order to Sqrt[2/Pi] Sin[z]/Sqrt[z]; SymPy keeps
    # besselj(1/2, z), which is DELIBERATE -- Rubi's rules pattern-match on
    # BesselJ[n_, a+b x], and auto-expanding would stop them matching. Same number:
    mma = sympy.sqrt(2 / sympy.pi) * sympy.sin(z) / sympy.sqrt(z)
    assert sympy.simplify(eager_BesselJ(sympy.Rational(1, 2), z) - mma) == 0


def test_eager_ExpIntegralE_matches_mathematica():
    from sympy_wolfram.functions_eager import eager_ExpIntegralE
    n, z = sympy.symbols('n z')
    assert eager_ExpIntegralE(1, z) == sympy.expint(1, z)  # stays symbolic
    assert eager_ExpIntegralE(n, z) == sympy.expint(n, z)
    # exact arguments must NOT be forced to a float (the old impl called .evalf())
    assert eager_ExpIntegralE(2, sympy.Rational(3, 2)) == sympy.expint(2, sympy.Rational(3, 2))
    # an inexact argument evaluates, as in MMA: ExpIntegralE[2, 1.5] == 0.0731007865384809
    assert abs(float(eager_ExpIntegralE(2, 1.5)) - 0.0731007865384809) < 1e-15


def test_eager_Root_matches_mathematica_indexing():
    """MMA's Root[poly,k] is 1-based and orders roots real-first-ascending, then
    complex. SymPy's CRootOf uses the same order but is 0-based, so k-1 is the whole
    translation. Root values below are Mathematica 12.2's."""
    from sympy_wolfram.functions_eager import eager_Root
    x = sympy.Symbol('x')
    assert sympy.simplify(eager_Root(x ** 2 - 2, 1) + sympy.sqrt(2)) == 0   # -Sqrt[2]
    assert sympy.simplify(eager_Root(x ** 2 - 2, 2) - sympy.sqrt(2)) == 0   # +Sqrt[2]
    # x^3-x-1: root 1 is the REAL one, roots 2/3 the conjugate pair (negative imag first)
    assert abs(complex(sympy.N(eager_Root(x ** 3 - x - 1, 1), 20)) - 1.324717957244746) < 1e-12
    r2 = complex(sympy.N(eager_Root(x ** 3 - x - 1, 2), 20))
    assert abs(r2 - (-0.662358978622373 - 0.5622795120623012j)) < 1e-12
    # x^4-1: reals ascending first
    assert abs(complex(sympy.N(eager_Root(x ** 4 - 1, 1), 20)) - (-1)) < 1e-12
    assert abs(complex(sympy.N(eager_Root(x ** 4 - 1, 2), 20)) - 1) < 1e-12
    assert abs(complex(sympy.N(eager_Root(x ** 2 + 1, 1), 20)) - (-1j)) < 1e-12


@pytest.mark.parametrize('name', ['Factorial', 'Zeta', 'PolyGamma', 'BesselJ',
                                  'ExpIntegralE', 'Root', 'Discriminant'])
def test_special_function_nodes_defer_then_evaluate(name):
    """Each is a MathematicaExpr: constructing it does nothing, doit() computes."""
    node = getattr(mf, name)
    z = sympy.Symbol('z')
    built = node(z, 2) if name in ('Zeta', 'PolyGamma', 'BesselJ', 'ExpIntegralE',
                                   'Root', 'Discriminant') else node(z)
    assert isinstance(built, mf.MathematicaExpr)
    built.doit()   # must not raise


# ---------------------------------------------------------------------------
# 5. ProductLog / Identity / ExpIntegralEi / LogIntegral
#    All Wolfram BUILTINS. Expected values from Mathematica 12.2.
# ---------------------------------------------------------------------------

def test_eager_ProductLog_argument_order_is_reversed_vs_sympy():
    """MMA takes the branch index FIRST (ProductLog[k, z]); SymPy takes it LAST
    (LambertW(z, k)). Getting this backwards is silent and wrong, so it is pinned."""
    from sympy_wolfram.functions_eager import eager_ProductLog
    z = sympy.Symbol('z')
    assert eager_ProductLog(z) == sympy.LambertW(z)
    assert eager_ProductLog(0) == 0                        # ProductLog[0] == 0
    assert eager_ProductLog(-1 / sympy.E) == -1            # ProductLog[-1/E] == -1
    # MMA keeps exact/symbolic input symbolic -- the old impl called .evalf()
    assert eager_ProductLog(1) == sympy.LambertW(1)
    # N[ProductLog[5.0]] == 1.3267246652422002
    assert abs(float(sympy.N(eager_ProductLog(5.0), 20)) - 1.3267246652422002) < 1e-15
    # branch k=-1: N[ProductLog[-1, -0.1]] == -3.577152063957297
    assert abs(float(sympy.N(eager_ProductLog(-1, -0.1), 20)) - (-3.577152063957297)) < 1e-14
    # branch k=0:  N[ProductLog[0, -0.1]]  == -0.11183255915896297
    assert abs(float(sympy.N(eager_ProductLog(0, -0.1), 20)) - (-0.11183255915896297)) < 1e-15


def test_eager_Identity():
    """Identity[z] == z. Rubi uses it to stop a coefficient folding away early:
    Int[-u_, x] := Identity[-1]*Int[u, x]."""
    from sympy_wolfram.functions_eager import eager_Identity, eager_Complex
    a, b = sympy.symbols('a b')
    assert eager_Identity(-1) == -1
    assert eager_Identity(0) == 0
    assert eager_Identity(a + b) == a + b
    # Complex[Identity[0], a] == Complex[0, a] == I a
    assert eager_Complex(eager_Identity(0), a) == sympy.I * a


def test_eager_ExpIntegralEi_and_LogIntegral():
    from sympy_wolfram.functions_eager import eager_ExpIntegralEi, eager_LogIntegral
    z = sympy.Symbol('z')
    assert eager_ExpIntegralEi(z) == sympy.Ei(z)           # stays symbolic
    assert eager_LogIntegral(z) == sympy.li(z)
    # N[ExpIntegralEi[1.0]] == 1.8951178163559368
    assert abs(float(sympy.N(eager_ExpIntegralEi(1.0), 20)) - 1.8951178163559368) < 1e-15
    # N[LogIntegral[2.0]] == 1.0451637801174924
    assert abs(float(sympy.N(eager_LogIntegral(2.0), 20)) - 1.0451637801174924) < 1e-14
    assert eager_LogIntegral(1) == -sympy.oo                # LogIntegral[1] == -Infinity


def test_Block_is_available_for_the_UseGamma_rule():
    """8.6 Gamma functions wraps its body in Block[{$UseGamma = True}, ...]; the rule
    now generates, so Block has to be importable where the rules look for it."""
    from rubi_integrate.utils import rubi_utils
    from sympy_wolfram.objects import Block, List, Set
    q = sympy.Symbol('q')
    assert rubi_utils.Block is Block
    # Block[{q = 1}, q + 2] == 3
    assert Block(List(Set(q, sympy.Integer(1))), q + 2).doit() == 3


# ---------------------------------------------------------------------------
# 6. Named Wolfram classes that evaluate to their SymPy equivalent
#
# ExpIntegralEi / LogIntegral / ProductLog / Identity are kept as NAMED classes
# rather than erased into a plain SymPy call by the code generator, so the Wolfram
# spelling stays readable at the call site and the translation lives in one place.
# They evaluate EAGERLY, so an expression built from them holds the SymPy object.
# https://reference.wolfram.com/language/ref/ExpIntegralEi.html
# https://reference.wolfram.com/language/ref/LogIntegral.html
# All values below are from Mathematica 12.2.
# ---------------------------------------------------------------------------

def test_ExpIntegralEi_node_evaluates_to_Ei():
    z = sympy.Symbol('z')
    assert isinstance(mf.ExpIntegralEi(z), MathematicaExpr)   # stays a Wolfram node
    assert mf.ExpIntegralEi(z).doit() == sympy.Ei(z)          # ExpIntegralEi[z]
    # N[ExpIntegralEi[1.0]] == 1.8951178163559368
    assert abs(float(sympy.N(mf.ExpIntegralEi(1.0).doit(), 20)) - 1.8951178163559368) < 1e-15


def test_LogIntegral_node_evaluates_to_li():
    z = sympy.Symbol('z')
    assert isinstance(mf.LogIntegral(z), MathematicaExpr)
    assert mf.LogIntegral(z).doit() == sympy.li(z)         # LogIntegral[z]
    assert mf.LogIntegral(1).doit() == -sympy.oo           # LogIntegral[1] == -Infinity
    # N[LogIntegral[2.0]] == 1.0451637801174924
    assert abs(float(sympy.N(mf.LogIntegral(2.0).doit(), 20)) - 1.0451637801174924) < 1e-14


def test_ProductLog_node_evaluates_to_LambertW_with_swapped_branch_index():
    """Why ProductLog keeps its own node: the correspondence is NOT identity.
    Mathematica's branch index comes FIRST, SymPy's LAST."""
    z, k = sympy.symbols('z k')
    assert isinstance(mf.ProductLog(z), MathematicaExpr)
    assert mf.ProductLog(z).doit() == sympy.LambertW(z)    # ProductLog[z]
    assert mf.ProductLog(0).doit() == 0                    # ProductLog[0] == 0
    assert mf.ProductLog(-1 / sympy.E).doit() == -1        # ProductLog[-1/E] == -1
    # the swap, on both branches:
    assert mf.ProductLog(k, z).doit() == sympy.LambertW(z, k)
    # N[ProductLog[-1, -0.1]] == -3.577152063957297
    assert abs(float(sympy.N(mf.ProductLog(-1, -0.1).doit(), 20)) - (-3.577152063957297)) < 1e-14
    # N[ProductLog[0, -0.1]]  == -0.11183255915896297
    assert abs(float(sympy.N(mf.ProductLog(0, -0.1).doit(), 20)) - (-0.11183255915896297)) < 1e-15
    # N[ProductLog[5.0]] == 1.3267246652422002
    assert abs(float(sympy.N(mf.ProductLog(5.0).doit(), 20)) - 1.3267246652422002) < 1e-15


def test_Identity_node():
    """Defined for interpreter completeness. The RULE generator replaces Identity[z]
    by z instead of emitting it -- it carries no meaning of its own."""
    a, b = sympy.symbols('a b')
    assert isinstance(mf.Identity(-1), MathematicaExpr)
    assert mf.Identity(-1).doit() == -1                    # Identity[-1] == -1
    assert mf.Identity(a + b).doit() == a + b


def test_nodes_are_importable_where_generated_code_expects_them():
    """Generated rule modules resolve these through `from rubi_utils import *`, and the
    generated TEST-SUITE modules through an explicit sympy_wolfram import."""
    from rubi_integrate.utils import rubi_utils
    for name in ('ExpIntegralEi', 'LogIntegral', 'ProductLog', 'Identity'):
        assert getattr(rubi_utils, name) is getattr(mf, name)


# ---------------------------------------------------------------------------
# 7. rewrite_as_standard_sympy() -- the Wolfram -> SymPy translation protocol
#
# Distinct from doit(): doit() EVALUATES with Mathematica semantics, this one only
# swaps the head for the SymPy equivalent and stops, so the result is still a
# function application whose arguments (wildcards, in a rule pattern) survive.
# ---------------------------------------------------------------------------

def test_rewrite_is_a_translation_not_an_evaluation():
    """The whole point: it must NOT compute. A pattern's arguments have to survive."""
    assert mf.Factorial(5).doit() == 120                     # doit computes
    assert mf.Factorial(5).rewrite_as_standard_sympy() == sympy.factorial(5, evaluate=False)
    assert mf.Factorial(5).rewrite_as_standard_sympy() != 120
    assert mf.Zeta(2).doit() == sympy.pi ** 2 / 6
    assert mf.Zeta(2).rewrite_as_standard_sympy().func is sympy.zeta


def test_Gamma_rewrite_dispatches_on_arity():
    """The case that motivated the protocol: Mathematica overloads Gamma, SymPy does
    not, so no name table can express it -- the node must inspect its own arity."""
    from sympy_wolfram.objects import Gamma
    a, z = sympy.symbols('a z')
    assert Gamma(a).rewrite_as_standard_sympy() == sympy.gamma(a)
    assert Gamma(a, z).rewrite_as_standard_sympy() == sympy.uppergamma(a, z)


def test_ProductLog_rewrite_moves_the_branch_index():
    z, k = sympy.symbols('z k')
    assert mf.ProductLog(z).rewrite_as_standard_sympy() == sympy.LambertW(z)
    # Mathematica ProductLog[k, z] == SymPy LambertW(z, k)
    assert mf.ProductLog(k, z).rewrite_as_standard_sympy() == sympy.LambertW(z, k)


def test_PolyGamma_one_argument_form_becomes_order_zero():
    z, n = sympy.symbols('z n')
    assert mf.PolyGamma(z).rewrite_as_standard_sympy() == sympy.polygamma(0, z)
    assert mf.PolyGamma(n, z).rewrite_as_standard_sympy() == sympy.polygamma(n, z)


@pytest.mark.parametrize('name, args, target', [
    ('BesselJ',       ('n', 'z'), sympy.besselj),
    ('ExpIntegralE',  ('n', 'z'), sympy.expint),
    ('ExpIntegralEi', ('z',),     sympy.Ei),
    ('LogIntegral',   ('z',),     sympy.li),
])
def test_one_to_one_rewrites(name, args, target):
    syms = sympy.symbols(' '.join(args))
    syms = (syms,) if not isinstance(syms, tuple) else syms
    assert getattr(mf, name)(*syms).rewrite_as_standard_sympy() == target(*syms)


def test_Identity_rewrite_drops_the_head_entirely():
    a = sympy.Symbol('a')
    assert mf.Identity(a).rewrite_as_standard_sympy() == a


def test_default_returns_self_when_there_is_no_sympy_equivalent():
    """Most nodes model Wolfram LANGUAGE constructs (With/Module/Condition/Set). For
    those, 'no standard equivalent' is a real answer, not an error -- which is why the
    base implementation returns self rather than raising."""
    from sympy_wolfram.objects import With, List, Set
    x = sympy.Symbol('x')
    node = With(List(Set(x, sympy.Integer(2))), x)
    assert node.rewrite_as_standard_sympy() is node


def test_recursive_helper_rewrites_nested_nodes():
    from sympy_wolfram.objects import Gamma, rewrite_as_standard_sympy
    a, z = sympy.symbols('a z')
    nested = sympy.log(Gamma(a)) + mf.ProductLog(z)
    assert rewrite_as_standard_sympy(nested) == sympy.log(sympy.gamma(a)) + sympy.LambertW(z)
    # non-Wolfram input is returned untouched
    assert rewrite_as_standard_sympy(a + z) == a + z


# ---------------------------------------------------------------------------
# 8. Numeric cross-check of rewrite_as_standard_sympy() against real Mathematica
# ---------------------------------------------------------------------------

# Every entry below is (Wolfram head, exact args, Re, Im) where the two floats are
# what REAL Mathematica returned for N[head[args], 25] -- captured by running the
# expressions through wolframscript, not by trusting the documentation.
#
# Structural agreement is not enough for these heads. The translation reorders
# arguments (ProductLog[k, z] -> LambertW(z, k)), changes arity (PolyGamma[z] ->
# polygamma(0, z)) and picks between same-family functions (Gamma[a, z] is the
# UPPER incomplete gamma, so uppergamma and never lowergamma). Each of those is a
# silent wrong-answer bug that type checks and matches fine -- only numbers catch it.
# The points were chosen to be discriminating: negative and complex arguments, and
# for ProductLog the k = -1 branch, where swapping the arguments gives a different
# finite value rather than an error.
MATHEMATICA_REFERENCE_VALUES = [
    ('Gamma', ('37/10',), 4.170651783796603, 0.0),
    ('Gamma', ('-5/2',), -0.9453087204829419, 0.0),
    ('Gamma', ('1/2 + 6*I/5',), 0.22298482861259625, -0.30830839880793004),
    ('Gamma', ('23/10', '17/10',), 0.6803740490674334, 0.0),
    ('Gamma', ('1/2', '3',), 0.025356509323463443, 0.0),
    ('Gamma', ('-3/2', '2',), 0.011832994103345998, 0.0),
    ('Gamma', ('1 + I', '2 - I/2',), 0.002290564090173998, 0.1560931329893612),
    ('Gamma', ('5', '1/10',), 23.999998159727596, 0.0),
    ('BesselJ', ('0', '5/2',), -0.048383776468198, 0.0),
    ('BesselJ', ('3/2', '16/5',), 0.43713398386173985, 0.0),
    ('BesselJ', ('2', '-13/10',), 0.18302669876873764, 0.0),
    ('BesselJ', ('1/2', '1 + I',), 0.9679012828901307, 0.060204606214281704),
    ('BesselJ', ('-1', '7/3',), -0.5337007898361258, 0.0),
    ('ExpIntegralE', ('1', '2',), 0.04890051070806112, 0.0),
    ('ExpIntegralE', ('5/2', '7/10',), 0.19522126482361293, 0.0),
    ('ExpIntegralE', ('0', '3/2',), 0.14875344009895322, 0.0),
    ('ExpIntegralE', ('1', '1 + I',), 0.00028162445198141834, -0.17932453503935894),
    ('ExpIntegralE', ('3', '1/4',), 0.32468412597814367, 0.0),
    ('ExpIntegralEi', ('3/2',), 3.301285449129798, 0.0),
    ('ExpIntegralEi', ('-2',), -0.04890051070806112, 0.0),
    ('ExpIntegralEi', ('3/10 + 11*I/10',), 0.6809884385414778, 2.4979824731744684),
    ('ExpIntegralEi', ('1/20',), -2.3678845985793746, 0.0),
    ('LogIntegral', ('2',), 1.045163780117493, 0.0),
    ('LogIntegral', ('1/2',), -0.37867104306108795, 0.0),
    ('LogIntegral', ('37/10',), 2.7449413517896906, 0.0),
    ('LogIntegral', ('3/2 + I',), 0.9555492098621429, 1.5677515696641124),
    ('Factorial', ('5',), 120.0, 0.0),
    ('Factorial', ('1/2',), 0.886226925452758, 0.0),
    ('Factorial', ('-1/2',), 1.772453850905516, 0.0),
    ('Factorial', ('0',), 1.0, 0.0),
    ('Factorial', ('21/10',), 2.197620278392477, 0.0),
    ('PolyGamma', ('5/2',), 0.7031566406452432, 0.0),
    ('PolyGamma', ('-3/2',), 0.7031566406452432, 0.0),
    ('PolyGamma', ('1 + I',), 0.09465032062247698, 1.0766740474685812),
    ('PolyGamma', ('1', '5/2',), 0.49035775610023485, 0.0),
    ('PolyGamma', ('2', '7/10',), -6.434992874190923, 0.0),
    ('PolyGamma', ('3', '6/5',), 3.24499486472578, 0.0),
    ('PolyGamma', ('0', '1/3',), -3.1320337800208065, 0.0),
    ('ProductLog', ('1',), 0.5671432904097838, 0.0),
    ('ProductLog', ('-1/5',), -0.25917110181907377, 0.0),
    ('ProductLog', ('2 + I',), 0.8906840692020068, 0.22072564954715954),
    ('ProductLog', ('1/100',), 0.009901473843595012, 0.0),
    ('ProductLog', ('-1', '-1/5',), -2.5426413577735265, 0.0),
    ('ProductLog', ('0', '2',), 0.8526055020137255, 0.0),
    ('ProductLog', ('-1', '-1/10',), -3.577152063957297, 0.0),
    ('ProductLog', ('-1', '-1/4',), -2.15329236411035, 0.0),
    ('Zeta', ('5/2',), 1.341487257250917, 0.0),
    ('Zeta', ('-3/2',), -0.025485201889833036, 0.0),
    ('Zeta', ('1/2 + 14*I',), 0.02224114260999359, -0.10325812326645006),
    ('Zeta', ('5/2', '13/10',), 0.7832185539082374, 0.0),
    ('Zeta', ('3', '7/10',), 3.2174964370954613, 0.0),
    ('Zeta', ('2', '1/4',), 17.19732915450711, 0.0),
    ('Identity', ('37/10',), 3.7, 0.0),
    ('Identity', ('1 + I',), 1.0, 1.0),
]


@pytest.mark.parametrize('head, args, expected_re, expected_im', MATHEMATICA_REFERENCE_VALUES)
def test_rewrite_matches_mathematica_numerically(head, args, expected_re, expected_im):
    """The rewritten SymPy expression must evaluate to what Mathematica evaluates to."""
    import sympy_wolfram.objects as _objects
    node = getattr(mf, head, None) or getattr(_objects, head)
    sargs = [sympy.sympify(a) for a in args]
    rewritten = node(*sargs).rewrite_as_standard_sympy()
    value = sympy.sympify(sympy.N(rewritten, 25))
    got_re, got_im = float(sympy.re(value)), float(sympy.im(value))
    assert math.isfinite(got_re) and math.isfinite(got_im), (
        f'{head}{args} rewrote to a non-finite value: {rewritten}')
    for got, want, part in ((got_re, expected_re, 'Re'), (got_im, expected_im, 'Im')):
        assert abs(got - want) <= 1e-9 * max(1.0, abs(got), abs(want)), (
            f'{part} of {head}{args} -> {rewritten}: got {got!r}, Mathematica gives {want!r}')


def test_polynomial_remainder_nonmonomial_small_denominator():
    """The fraction-field (Laurent) reduction must work for SMALL non-monomial
    denominators: Rubi's 1.1.2.8 #86 With-block computes
    PolynomialQuotient[x^3/(c+d x), a+b x^2, x], and refusing the reduction there
    returned garbage e/f coefficients -- Int[x^3/((a+x)(b^2+x^2)^2)] (and every
    integral reduced to it, e.g. tanh^3/(a+b sinh)) came back WRONG. The blow-up
    guard (§42) must only refuse LARGE non-monomial denominators (the nonsense
    candidate bindings commutative enumeration feeds guards, e.g. deg-8 den vs a
    symbolic quartic)."""
    from sympy_wolfram.functions_eager import eager_PolynomialRemainder
    xx, a, b, c, d = sympy.symbols('x a b c d')
    # small non-monomial den: full reduction mod a+b x^2 -> a LINEAR e + f*x form
    r = eager_PolynomialRemainder(xx ** 3 / (c + d * xx), a + b * xx ** 2, xx)
    assert sympy.degree(sympy.fraction(sympy.together(r))[0], xx) <= 1
    assert not sympy.fraction(sympy.together(r))[1].has(xx)
    # large non-monomial den (the §42 bomb shape): returned unreduced, instantly
    p_bomb = (xx ** 4 - 5 * xx ** 2 + 4) ** -2
    q_bomb = -d - c * xx + d * xx * (xx ** 3 - 2 * xx ** 2 + 2)
    assert eager_PolynomialRemainder(p_bomb, q_bomb, xx) == p_bomb


def test_replaceall_accepts_a_list_of_rules():
    """Mathematica's ReplaceAll takes either one Rule or a LIST of rules applied
    simultaneously. The list branch used to return the expression UNCHANGED, so
    the Module[{aa,bb,cc}, ... ReplaceAll[..., {aa->a, bb->b, cc->c}]] idiom
    (1.2.2.3 #86 and friends) leaked its scoped dummies into antiderivatives."""
    aa, bb = sympy.symbols('aa bb')
    e = aa ** 2 + bb
    assert mf.ReplaceAll(e, mf.Rule(aa, Integer(3))).doit() == 9 + bb
    out = mf.ReplaceAll(e, mf.List(mf.Rule(aa, Integer(3)), mf.Rule(bb, Integer(4)))).doit()
    assert out == 13
    # simultaneous, not sequential: aa->bb, bb->aa swaps rather than chains
    swapped = mf.ReplaceAll(aa - bb, mf.List(mf.Rule(aa, bb), mf.Rule(bb, aa))).doit()
    assert swapped == bb - aa


@pytest.mark.parametrize('imin, imax, want', [
    # Mathematica steps the iterator by 1 starting EXACTLY at imin and stops at the
    # last value <= imax: only the UPPER bound truncates. Values read off
    # Mathematica 12.2 (RUBI_PORT_DEFECTS.md 50) -- rounding imin up gives 6 for the
    # half-integer case, which is what this port used to return.
    (0, sympy.Rational(11, 3), 6),          # k = 0,1,2,3
    (sympy.Rational(1, 2), sympy.Rational(7, 2), 8),   # k = 1/2,3/2,5/2,7/2
    (0, 3, 6),
    (1, 4, 10),
    (0, -1, 0),                             # empty range
])
def test_Sum_iterates_like_mathematica(imin, imax, want):
    k = Symbol('k')
    got = mf.Sum(k, mf.List(k, imin, imax)).doit()
    assert got == Integer(want)


def test_Sum_fractional_upper_bound_expands():
    """Mathematica: Sum[x^(3k), {k, 0, 11/3}] = 1 + x^3 + x^6 + x^9.

    sympy.Sum leaves a fractional-bound sum UNEVALUATED, which is how a Rubi rule
    ended up carrying an unexpanded Sum (and its loose iterator) into a result.
    """
    x, k = Symbol('x'), Symbol('k')
    got = mf.Sum(x**(3*k), mf.List(k, 0, sympy.Rational(11, 3))).doit()
    assert sympy.expand(got - (1 + x**3 + x**6 + x**9)) == 0
    assert k not in got.free_symbols

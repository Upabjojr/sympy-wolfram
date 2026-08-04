# -*- coding: utf-8 -*-
"""Tests for the full Mathematica-string → FFL → SymPy pipeline.

Verifies that ``mathematica_to_ffl`` produces correct FFL structure,
``mathematica_to_sympy_code`` yields eval-able code strings,
``mathematica_to_sympy_short_code`` yields simplified code strings, and
``mathematica_to_sympy`` yields the expected SymPy expression objects.

Uses pytest parametrize with a single unified case list to avoid
repeating the same expressions across multiple test classes.
"""
import pytest
import sympy
from sympy import (
    Symbol, Integer, Rational, log, sqrt, sin, cos, tan, exp,
    simplify, pi, oo, S, atan, asin, acos, Function,
)

from sympy_wolfram.interpreter import (
    mathematica_to_ffl,
    mathematica_to_sympy_code,
    mathematica_to_sympy_short_code,
    mathematica_to_sympy,
    ffl_to_sympy_code,
    ffl_to_sympy_short_code,
)

from sympy_matching.wild import WildSymbol, IDENTITY_ELEMENT

from sympy_wolfram.parser import parse_mathematica_to_fullformlist

# Parsing Mathematica SOURCE STRINGS needs sympy's parse_mathematica_to_fullformlist,
# which released sympy versions (<= 1.14) do not ship. Tests that start from an FFL
# list (TestFFLToSympyCode) run everywhere; string-input tests skip on such sympy.
requires_mathematica_parser = pytest.mark.skipif(
    parse_mathematica_to_fullformlist is None,
    reason='sympy.parsing.mathematica.parse_mathematica_to_fullformlist not '
           'available in this sympy version (needs sympy > 1.14)')

# These tests exercise the generic converter, which has no notion of a
# distinguished variable -- the CALLER decides which names are externally
# bound. The cases below are written around ``x``, so they declare it.
_X = {'x': 'x'}


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

x = Symbol('x')
a, b, m, n = Symbol('a'), Symbol('b'), Symbol('m'), Symbol('n')
SYMS = {'a': a, 'b': b, 'm': m, 'n': n}


def assert_sympy_equal(result, expected):
    """Assert two SymPy objects are equal, handling oo/atoms gracefully."""
    if result == expected:
        return
    assert simplify(result - expected) == 0


def _assert_wildsymbol(obj, name, is_optional):
    """Assert obj is a WildSymbol with given name and optionality."""
    assert isinstance(obj, WildSymbol), (
        f"Expected WildSymbol, got {type(obj).__name__}: {obj!r}"
    )
    assert obj.wildcard_name == name, (
        f"Expected wildcard_name={name!r}, got {obj.wildcard_name!r}"
    )
    if is_optional:
        assert obj.optional_value is IDENTITY_ELEMENT, (
            f"Expected optional_value=IDENTITY_ELEMENT, got {obj.optional_value!r}"
        )
    else:
        assert obj.optional_value is None, (
            f"Expected optional_value=None, got {obj.optional_value!r}"
        )


def _collect_wildsymbols(expr):
    """Recursively collect all WildSymbol instances from a SymPy expression.

    Returns them sorted by wildcard_name for order-independent comparison
    (SymPy may reorder args in commutative operations like Add/Mul).
    """
    if isinstance(expr, WildSymbol):
        return [expr]
    result = []
    if hasattr(expr, 'args'):
        for arg in expr.args:
            result.extend(_collect_wildsymbols(arg))
    result.sort(key=lambda ws: ws.wildcard_name)
    return result


def _expected_defs_from_wildcards(expected_wildcards):
    """Compute expected wild_defs list from expected_wildcards tuples."""
    if not expected_wildcards:
        return []
    defs = []
    for name, is_opt in expected_wildcards:
        if is_opt:
            defs.append(
                f"_{name}_ = WildSymbol('{name}', optional_value=IDENTITY_ELEMENT)"
            )
        else:
            defs.append(f"{name}_ = WildSymbol('{name}')")
    return defs


# ---------------------------------------------------------------------------
# Unified test data
#
# Each entry:
#   (mma_expr, expected_ffl, expected_code, expected_short_code,
#    expected_sympy, extra_symbols, expected_wildcards)
#
# expected_short_code: the output of mathematica_to_sympy_short_code — i.e.
#   str(eval(code)) when eval(str(eval(code))) round-trips successfully,
#   otherwise the same as expected_code.
# namespace=None means no extra symbols needed for eval.
# expected_wildcards=None means no WildSymbol checking; when set to a list
#   of (name, is_optional) tuples, the eval result is checked for matching
#   WildSymbol instances instead of using assert_sympy_equal.
# When expected_wildcards is provided, expected_sympy may be None (since
#   WildSymbol instances have unique indices and cannot be compared with ==).
# ---------------------------------------------------------------------------

CASES = [
    # --- atoms ---
    (
        "x",
        "x",
        "x",
        "x",
        x,
        None,
        None,
    ),
    (
        "42",
        "42",
        "Integer(42)",
        "Integer(42)",  # str(Integer(42))="42" but eval("42") is int, not Basic
        Integer(42),
        None,
        None,
    ),
    (
        "Pi",
        "Pi",
        "sympy.pi",
        "pi",
        pi,
        None,
        None,
    ),
    (
        "E",
        "E",
        "sympy.E",
        "sympy.E",  # str(E)="E" but bare "E" not in eval namespace
        sympy.E,
        None,
        None,
    ),
    (
        "Infinity",
        "Infinity",
        "sympy.oo",
        "oo",
        oo,
        None,
        None,
    ),
    # --- arithmetic ---
    (
        "x^2",
        ['Power', 'x', '2'],
        "(x)**(Integer(2))",
        "x**2",
        x**2,
        None,
        None,
    ),
    (
        "3/4",
        ['Times', '3', ['Power', '4', '-1']],
        "(Integer(3) * (Integer(4))**(Integer(-1)))",
        "sympy.S(3)/4",  # SymPy 1.14+: Rational(3,4) prints as sympy.S(3)/4
        Rational(3, 4),
        None,
        None,
    ),
    (
        "x + 1",
        ['Plus', 'x', '1'],
        "(x + Integer(1))",
        "x + 1",
        x + 1,
        None,
        None,
    ),
    (
        "x - 1",
        ['Plus', 'x', '-1'],
        "(x + Integer(-1))",
        "x - 1",
        x - 1,
        None,
        None,
    ),
    (
        "-x",
        ['Times', '-1', 'x'],
        "(Integer(-1) * x)",
        "-x",
        -x,
        None,
        None,
    ),
    (
        "1/x",
        ['Times', '1', ['Power', 'x', '-1']],
        "(x)**(Integer(-1))",
        "1/x",
        1/x,
        None,
        None,
    ),
    (
        "3*x",
        ['Times', '3', 'x'],
        "(Integer(3) * x)",
        "3*x",
        3*x,
        None,
        None,
    ),
    (
        "a + b",
        ['Plus', 'a', 'b'],
        "(Symbol('a') + Symbol('b'))",
        "a + b",
        a + b,
        SYMS,
        None,
    ),
    (
        "a*b",
        ['Times', 'a', 'b'],
        "(Symbol('a') * Symbol('b'))",
        "a*b",
        a * b,
        SYMS,
        None,
    ),
    (
        "Pi*x",
        ['Times', 'Pi', 'x'],
        "(sympy.pi * x)",
        "pi*x",
        pi * x,
        None,
        None,
    ),
    # --- functions ---
    ("Sin[x]", ['Sin', 'x'], "sympy.sin(x)", "sin(x)", sin(x), None, None),
    ("Cos[x]", ['Cos', 'x'], "sympy.cos(x)", "cos(x)", cos(x), None, None),
    ("Tan[x]", ['Tan', 'x'], "sympy.tan(x)", "tan(x)", tan(x), None, None),
    ("Log[x]", ['Log', 'x'], "sympy.log(x)", "log(x)", log(x), None, None),
    ("Exp[x]", ['Exp', 'x'], "sympy.exp(x)", "exp(x)", exp(x), None, None),
    ("Sqrt[x]", ['Sqrt', 'x'], "sympy.sqrt(x)", "sqrt(x)", sqrt(x), None, None),
    ("ArcTan[x]", ['ArcTan', 'x'], "sympy.atan(x)", "atan(x)", atan(x), None, None),
    ("ArcSin[x]", ['ArcSin', 'x'], "sympy.asin(x)", "asin(x)", asin(x), None, None),
    ("ArcCos[x]", ['ArcCos', 'x'], "sympy.acos(x)", "acos(x)", acos(x), None, None),
    (
        "Sin[Cos[x]]",
        ['Sin', ['Cos', 'x']],
        "sympy.sin(sympy.cos(x))",
        "sin(cos(x))",
        sin(cos(x)),
        None,
        None,
    ),
    (
        "Exp[Log[x]]",
        ['Exp', ['Log', 'x']],
        "sympy.exp(sympy.log(x))",
        "x",
        x,
        None,
        None,
    ),
    # --- composite / compound ---
    (
        "Exp[-x^2]",
        ['Exp', ['Times', '-1', ['Power', 'x', '2']]],
        "sympy.exp((Integer(-1) * (x)**(Integer(2))))",
        "exp(-x**2)",
        exp(-x**2),
        None,
        None,
    ),
    (
        "Sin[x]^2 + Cos[x]^2",
        ['Plus', ['Power', ['Sin', 'x'], '2'], ['Power', ['Cos', 'x'], '2']],
        "((sympy.sin(x))**(Integer(2)) + (sympy.cos(x))**(Integer(2)))",
        "sin(x)**2 + cos(x)**2",
        sin(x)**2 + cos(x)**2,
        None,
        None,
    ),
    (
        "x^(1/2)",
        ['Power', 'x', ['Times', '1', ['Power', '2', '-1']]],
        "(x)**((Integer(2))**(Integer(-1)))",
        "sqrt(x)",
        sqrt(x),
        None,
        None,
    ),
    (
        "x^(1/3)",
        ['Power', 'x', ['Times', '1', ['Power', '3', '-1']]],
        "(x)**((Integer(3))**(Integer(-1)))",
        "x**(sympy.S(1)/3)",  # SymPy 1.14+: Rational exponent prints as sympy.S(1)/3
        x**Rational(1, 3),
        None,
        None,
    ),
    (
        "x^3 + 3*x^2 + 3*x + 1",
        ['Plus', ['Power', 'x', '3'], ['Times', '3', ['Power', 'x', '2']],
         ['Times', '3', 'x'], '1'],
        "((x)**(Integer(3)) + (Integer(3) * (x)**(Integer(2)))"
        " + (Integer(3) * x) + Integer(1))",
        "x**3 + 3*x**2 + 3*x + 1",
        x**3 + 3*x**2 + 3*x + 1,
        None,
        None,
    ),
    (
        "(1 + 2*x)^3/6",
        ['Times', ['Power', ['Plus', '1', ['Times', '2', 'x']], '3'],
         ['Power', '6', '-1']],
        "(((Integer(1) + (Integer(2) * x)))**(Integer(3))"
        " * (Integer(6))**(Integer(-1)))",
        "(2*x + 1)**3/6",
        (1 + 2*x)**3 / 6,
        None,
        None,
    ),
    (
        "(x^2 - 1)/(x - 1)",
        ['Times', ['Plus', ['Power', 'x', '2'], '-1'],
         ['Power', ['Plus', 'x', '-1'], '-1']],
        "(((x)**(Integer(2)) + Integer(-1))"
        " * ((x + Integer(-1)))**(Integer(-1)))",
        "(x**2 - 1)/(x - 1)",
        (x**2 - 1) / (x - 1),
        None,
        None,
    ),
    # --- symbolic parameters ---
    (
        "(a + b*x)^m",
        ['Power', ['Plus', 'a', ['Times', 'b', 'x']], 'm'],
        "((Symbol('a') + (Symbol('b') * x)))**(Symbol('m'))",
        "(a + b*x)**m",
        (a + b*x)**m,
        SYMS,
        None,
    ),
    (
        "Log[a + b*x]",
        ['Log', ['Plus', 'a', ['Times', 'b', 'x']]],
        "sympy.log((Symbol('a') + (Symbol('b') * x)))",
        "log(a + b*x)",
        log(a + b*x),
        SYMS,
        None,
    ),
    (
        "Log[a + b*x]/b",
        ['Times', ['Log', ['Plus', 'a', ['Times', 'b', 'x']]],
         ['Power', 'b', '-1']],
        "(sympy.log((Symbol('a') + (Symbol('b') * x)))"
        " * (Symbol('b'))**(Integer(-1)))",
        "log(a + b*x)/b",
        log(a + b*x) / b,
        SYMS,
        None,
    ),
    (
        "(a + b*x)^(m + 1) / (b*(m + 1))",
        ['Times', ['Power', ['Plus', 'a', ['Times', 'b', 'x']],
                   ['Plus', 'm', '1']],
         ['Power', ['Times', 'b', ['Plus', 'm', '1']], '-1']],
        "(((Symbol('a') + (Symbol('b') * x)))**((Symbol('m') + Integer(1)))"
        " * ((Symbol('b') * (Symbol('m') + Integer(1))))**(Integer(-1)))",
        "(a + b*x)**(m + 1)/(b*(m + 1))",
        (a + b*x)**(m + 1) / (b * (m + 1)),
        SYMS,
        None,
    ),
    (
        "-1/(b*(a + b*x))",
        ['Times', '-1', ['Power', ['Times', 'b',
                                   ['Plus', 'a', ['Times', 'b', 'x']]], '-1']],
        "(Integer(-1) * ((Symbol('b')"
        " * (Symbol('a') + (Symbol('b') * x))))**(Integer(-1)))",
        "-1/(b*(a + b*x))",
        -1 / (b * (a + b*x)),
        SYMS,
        None,
    ),
    (
        "x^(n + 1)/(n + 1)",
        ['Times', ['Power', 'x', ['Plus', 'n', '1']],
         ['Power', ['Plus', 'n', '1'], '-1']],
        "((x)**((Symbol('n') + Integer(1)))"
        " * ((Symbol('n') + Integer(1)))**(Integer(-1)))",
        "x**(n + 1)/(n + 1)",
        x**(n + 1) / (n + 1),
        SYMS,
        None,
    ),
    # --- Pattern / Optional (WildSymbol) ---
    (
        "m_",
        ['Pattern', 'm', ['Blank']],
        "m_",
        "m_",
        None,
        None,
        [('m', False)],
    ),
    (
        "m_.",
        ['Optional', ['Pattern', 'm', ['Blank']]],
        "_m_",
        "_m_",
        None,
        None,
        [('m', True)],
    ),
    (
        "a_ + 1",
        ['Plus', ['Pattern', 'a', ['Blank']], '1'],
        "(a_ + Integer(1))",
        "a_ + 1",
        None,
        None,
        [('a', False)],
    ),
    (
        "a_ + b_*x",
        ['Plus', ['Pattern', 'a', ['Blank']],
         ['Times', ['Pattern', 'b', ['Blank']], 'x']],
        "(a_ + (b_ * x))",
        "x*b_ + a_",
        None,
        None,
        [('a', False), ('b', False)],
    ),
    (
        "a_. + b_.*x",
        ['Plus', ['Optional', ['Pattern', 'a', ['Blank']]],
         ['Times', ['Optional', ['Pattern', 'b', ['Blank']]], 'x']],
        "(_a_ + (_b_ * x))",
        "x*_b_ + _a_",
        None,
        None,
        [('a', True), ('b', True)],
    ),
    (
        "a_^m_",
        ['Power', ['Pattern', 'a', ['Blank']], ['Pattern', 'm', ['Blank']]],
        "(a_)**(m_)",
        "a_**m_",
        None,
        None,
        [('a', False), ('m', False)],
    ),
    (
        "a_.^2",
        ['Power', ['Optional', ['Pattern', 'a', ['Blank']]], '2'],
        "(_a_)**(Integer(2))",
        "_a_**2",
        None,
        None,
        [('a', True)],
    ),
    (
        "Sin[m_]",
        ['Sin', ['Pattern', 'm', ['Blank']]],
        "sympy.sin(m_)",
        "sin(m_)",
        None,
        None,
        [('m', False)],
    ),
    (
        "a_*x + b",
        ['Plus', ['Times', ['Pattern', 'a', ['Blank']], 'x'], 'b'],
        "((a_ * x) + Symbol('b'))",
        "b + x*a_",
        None,
        None,
        [('a', False)],
    ),
    (
        "x_",
        ['Pattern', 'x', ['Blank']],
        "x",
        "x",
        x,  # x_ resolves to x (fixed var), not a wildcard
        None,
        None,
    ),
]

CASE_IDS = [c[0] for c in CASES]


# ---------------------------------------------------------------------------
# Tests for mathematica_to_ffl
# ---------------------------------------------------------------------------

@requires_mathematica_parser
class TestMathematicaToFFL:
    """Verify FFL structure from Mathematica notation strings."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_ffl_structure(self, mma_expr, expected_ffl, expected_code,
                           expected_short_code, expected_sympy, extra_symbols,
                           expected_wildcards):
        assert mathematica_to_ffl(mma_expr) == expected_ffl


# ---------------------------------------------------------------------------
# Tests for mathematica_to_sympy_code — string + eval
# ---------------------------------------------------------------------------

@requires_mathematica_parser
class TestMathematicaToSympyCode:
    """Verify code strings and their eval results."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_code_string(self, mma_expr, expected_ffl, expected_code,
                         expected_short_code, expected_sympy, extra_symbols,
                         expected_wildcards):
        """The generated code string matches the expected output."""
        code, _defs, _symbols = mathematica_to_sympy_code(mma_expr, reserved_symbols=_X)
        assert code == expected_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_code_eval(self, mma_expr, expected_ffl, expected_code,
                       expected_short_code, expected_sympy, extra_symbols,
                       expected_wildcards):
        """eval(code, ns) produces the expected SymPy expression."""
        ns = {}
        code, wild_defs, _symbols = mathematica_to_sympy_code(mma_expr, namespace=ns, reserved_symbols=_X)
        if extra_symbols:
            ns.update(extra_symbols)
        result = eval(code, ns)
        if expected_wildcards is not None:
            wilds = _collect_wildsymbols(result)
            assert len(wilds) == len(expected_wildcards), (
                f"Expected {len(expected_wildcards)} WildSymbol(s), "
                f"got {len(wilds)}: {wilds}"
            )
            for ws, (name, is_opt) in zip(wilds, expected_wildcards):
                _assert_wildsymbol(ws, name, is_opt)
        elif expected_sympy is not None:
            assert_sympy_equal(result, expected_sympy)

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_wild_defs(self, mma_expr, expected_ffl, expected_code,
                       expected_short_code, expected_sympy, extra_symbols,
                       expected_wildcards):
        """wild_defs contains the correct variable definitions."""
        _code, wild_defs, _symbols = mathematica_to_sympy_code(mma_expr, reserved_symbols=_X)
        if expected_wildcards:
            expected_defs = _expected_defs_from_wildcards(expected_wildcards)
            assert sorted(wild_defs) == sorted(expected_defs)
        else:
            assert wild_defs == []

    # --- structural checks ---

    def test_returns_triple(self):
        result = mathematica_to_sympy_code("x", reserved_symbols=_X)
        assert isinstance(result, tuple) and len(result) == 3

    def test_code_is_string(self):
        code, _defs, _symbols = mathematica_to_sympy_code("x^2", reserved_symbols=_X)
        assert isinstance(code, str)

    def test_ns_contains_sympy(self):
        ns = {}
        _, _defs, _symbols = mathematica_to_sympy_code("x^2", namespace=ns, reserved_symbols=_X)
        assert isinstance(ns, dict)
        assert 'sympy' in ns

    def test_code_contains_no_ffl_heads(self):
        code, _defs, _symbols = mathematica_to_sympy_code("Sin[x] + Cos[x]^2", reserved_symbols=_X)
        assert 'Plus' not in code
        assert 'Power' not in code

    def test_code_uses_sympy_namespace(self):
        code, _defs, _symbols = mathematica_to_sympy_code("Sin[x]", reserved_symbols=_X)
        assert 'sympy.sin' in code

    def test_a_reserved_name_other_than_x(self):
        ns = {}
        code, _defs, _symbols = mathematica_to_sympy_code("t^2", reserved_symbols={'t': 'x'}, namespace=ns)
        assert 'x' in code
        result = eval(code, ns)
        assert result == Symbol('t')**2


# ---------------------------------------------------------------------------
# Tests for mathematica_to_sympy_short_code
# ---------------------------------------------------------------------------

@requires_mathematica_parser
class TestMathematicaToSympyShortCode:
    """Verify simplified code strings and their eval results."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_short_code_string(self, mma_expr, expected_ffl, expected_code,
                               expected_short_code, expected_sympy,
                               extra_symbols, expected_wildcards):
        """The simplified code string matches the expected output."""
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = mathematica_to_sympy_short_code(
            mma_expr, namespace=ns, reserved_symbols=_X)
        assert short == expected_short_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_short_code_eval(self, mma_expr, expected_ffl, expected_code,
                             expected_short_code, expected_sympy,
                             extra_symbols, expected_wildcards):
        """eval(short_code, ns) produces the expected SymPy expression."""
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = mathematica_to_sympy_short_code(
            mma_expr, namespace=ns, reserved_symbols=_X)
        result = eval(short, ns)
        if expected_wildcards is not None:
            wilds = _collect_wildsymbols(result)
            assert len(wilds) == len(expected_wildcards)
            for ws, (name, is_opt) in zip(wilds, expected_wildcards):
                _assert_wildsymbol(ws, name, is_opt)
        elif expected_sympy is not None:
            assert_sympy_equal(result, expected_sympy)

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_short_code_wild_defs(self, mma_expr, expected_ffl, expected_code,
                                  expected_short_code, expected_sympy,
                                  extra_symbols, expected_wildcards):
        """wild_defs from short_code matches the expected variable definitions."""
        ns = dict(extra_symbols or {})
        _short, wild_defs, _symbols = mathematica_to_sympy_short_code(
            mma_expr, namespace=ns, reserved_symbols=_X)
        if expected_wildcards:
            expected_defs = _expected_defs_from_wildcards(expected_wildcards)
            assert sorted(wild_defs) == sorted(expected_defs)
        else:
            assert wild_defs == []


# ---------------------------------------------------------------------------
# Tests for mathematica_to_sympy (end-to-end)
# ---------------------------------------------------------------------------

@requires_mathematica_parser
class TestMathematicaToSympy:
    """Full pipeline: Mathematica string → SymPy object."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_expression(self, mma_expr, expected_ffl, expected_code,
                        expected_short_code, expected_sympy, extra_symbols,
                        expected_wildcards):
        ns = dict(extra_symbols or {})
        result = mathematica_to_sympy(mma_expr, namespace=ns, reserved_symbols=_X)
        if expected_wildcards is not None:
            wilds = _collect_wildsymbols(result)
            assert len(wilds) == len(expected_wildcards)
            for ws, (name, is_opt) in zip(wilds, expected_wildcards):
                _assert_wildsymbol(ws, name, is_opt)
        elif expected_sympy is not None:
            assert_sympy_equal(result, expected_sympy)

    # --- additional cases ---

    def test_division_by_two(self):
        assert_sympy_equal(mathematica_to_sympy("x/2", reserved_symbols=_X), x/2)

    def test_x_cubed(self):
        assert mathematica_to_sympy("x^3", reserved_symbols=_X) == x**3

    def test_sqrt_linear(self):
        """2*(a + b*x)^(3/2)/(3*b)"""
        result = mathematica_to_sympy("2*(a + b*x)^(3/2)/(3*b)", namespace=SYMS, reserved_symbols=_X)
        expected = Rational(2, 3) * (a + b*x)**Rational(3, 2) / b
        assert_sympy_equal(result, expected)

    def test_pattern_is_wildsymbol(self):
        """m_ parses to a WildSymbol instance."""
        result = mathematica_to_sympy("m_", reserved_symbols=_X)
        assert isinstance(result, WildSymbol)
        assert result.wildcard_name == 'm'
        assert result.optional_value is None

    def test_optional_pattern_has_identity(self):
        """m_. parses to a WildSymbol with IDENTITY_ELEMENT."""
        result = mathematica_to_sympy("m_.", reserved_symbols=_X)
        assert isinstance(result, WildSymbol)
        assert result.wildcard_name == 'm'
        assert result.optional_value is IDENTITY_ELEMENT

    def test_reserved_name_pattern_is_the_plain_symbol(self):
        """x_ resolves to Symbol('x'), not a WildSymbol."""
        result = mathematica_to_sympy("x_", reserved_symbols=_X)
        assert result == Symbol('x')
        assert not isinstance(result, WildSymbol)


# ---------------------------------------------------------------------------
# Tests for ffl_to_sympy_code (FFL-level API)
# ---------------------------------------------------------------------------

class TestFFLToSympyCode:
    """Verify ffl_to_sympy_code produces same results as mathematica_to_sympy_code."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_code_string(self, mma_expr, expected_ffl, expected_code,
                         expected_short_code, expected_sympy, extra_symbols,
                         expected_wildcards):
        """ffl_to_sympy_code produces the same code as mathematica_to_sympy_code."""
        code, _defs, _symbols = ffl_to_sympy_code(expected_ffl, reserved_symbols=_X)
        assert code == expected_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_code_eval(self, mma_expr, expected_ffl, expected_code,
                       expected_short_code, expected_sympy, extra_symbols,
                       expected_wildcards):
        """eval(code, ns) from ffl_to_sympy_code produces correct SymPy expr."""
        ns = {}
        code, wild_defs, _symbols = ffl_to_sympy_code(expected_ffl, namespace=ns, reserved_symbols=_X)
        if extra_symbols:
            ns.update(extra_symbols)
        result = eval(code, ns)
        if expected_wildcards is not None:
            wilds = _collect_wildsymbols(result)
            assert len(wilds) == len(expected_wildcards)
            for ws, (name, is_opt) in zip(wilds, expected_wildcards):
                _assert_wildsymbol(ws, name, is_opt)
        elif expected_sympy is not None:
            assert_sympy_equal(result, expected_sympy)

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_wild_defs(self, mma_expr, expected_ffl, expected_code,
                       expected_short_code, expected_sympy, extra_symbols,
                       expected_wildcards):
        """wild_defs from ffl_to_sympy_code matches expected."""
        _code, wild_defs, _symbols = ffl_to_sympy_code(expected_ffl, reserved_symbols=_X)
        if expected_wildcards:
            expected_defs = _expected_defs_from_wildcards(expected_wildcards)
            assert sorted(wild_defs) == sorted(expected_defs)
        else:
            assert wild_defs == []


# ---------------------------------------------------------------------------
# Tests for ffl_to_sympy_short_code (FFL-level API)
# ---------------------------------------------------------------------------

class TestFFLToSympyShortCode:
    """Verify ffl_to_sympy_short_code produces same results as mathematica_to_sympy_short_code."""

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_short_code_string(self, mma_expr, expected_ffl, expected_code,
                               expected_short_code, expected_sympy,
                               extra_symbols, expected_wildcards):
        """ffl_to_sympy_short_code produces the same short code."""
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = ffl_to_sympy_short_code(
            expected_ffl, namespace=ns, reserved_symbols=_X)
        assert short == expected_short_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,expected_wildcards",
        CASES, ids=CASE_IDS,
    )
    def test_short_code_eval(self, mma_expr, expected_ffl, expected_code,
                             expected_short_code, expected_sympy,
                             extra_symbols, expected_wildcards):
        """eval(short_code, ns) from ffl_to_sympy_short_code is correct."""
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = ffl_to_sympy_short_code(
            expected_ffl, namespace=ns, reserved_symbols=_X)
        result = eval(short, ns)
        if expected_wildcards is not None:
            wilds = _collect_wildsymbols(result)
            assert len(wilds) == len(expected_wildcards)
            for ws, (name, is_opt) in zip(wilds, expected_wildcards):
                _assert_wildsymbol(ws, name, is_opt)
        elif expected_sympy is not None:
            assert_sympy_equal(result, expected_sympy)


# ---------------------------------------------------------------------------
# Custom function mapping tests
#
# Verifies the `custom_functions` parameter that allows users to map
# arbitrary Wolfram head names to custom SymPy-compatible callables.
#
# Each entry:
#   (mma_expr, expected_ffl, expected_code, expected_short_code,
#    expected_sympy, extra_symbols, custom_functions_dict)
# ---------------------------------------------------------------------------


# --- Define test custom functions/classes ---

import types

# A simple module-like namespace to simulate "my_module.FunctionCustom"
_custom_module = types.ModuleType("my_module")


class FunctionCustom(Function):
    """A custom SymPy Function subclass for testing."""
    pass


class SpecialOp(Function):
    """Another custom SymPy Function subclass for testing (two-arg)."""
    pass


# Attach to the fake module
_custom_module.FunctionCustom = FunctionCustom
_custom_module.SpecialOp = SpecialOp


# A standalone callable (no module prefix)
class DirectFunc(Function):
    """Custom function accessed without module prefix."""
    pass


# --- Custom functions dict variants used in test cases ---

CUSTOM_FUNCS_MODULE = {
    "FunctionCustom": ("my_module.FunctionCustom", _custom_module),
}

CUSTOM_FUNCS_SPECIAL = {
    "WolframSpecial": ("my_module.SpecialOp", _custom_module),
}

CUSTOM_FUNCS_DIRECT = {
    "Foo": ("DirectFunc", DirectFunc),
}

CUSTOM_FUNCS_MULTI = {
    "FunctionCustom": ("my_module.FunctionCustom", _custom_module),
    "WolframSpecial": ("my_module.SpecialOp", _custom_module),
    "Foo": ("DirectFunc", DirectFunc),
}


CUSTOM_CASES = [
    ("FunctionCustom[x]", ['FunctionCustom', 'x'],
     "my_module.FunctionCustom(x)", "my_module.FunctionCustom(x)",
     FunctionCustom(x), None, CUSTOM_FUNCS_MODULE),
    ("FunctionCustom[x^2 + 1]",
     ['FunctionCustom', ['Plus', ['Power', 'x', '2'], '1']],
     "my_module.FunctionCustom(((x)**(Integer(2)) + Integer(1)))",
     "my_module.FunctionCustom(((x)**(Integer(2)) + Integer(1)))",
     FunctionCustom(x**2 + 1), None, CUSTOM_FUNCS_MODULE),
    ("WolframSpecial[x]", ['WolframSpecial', 'x'],
     "my_module.SpecialOp(x)", "my_module.SpecialOp(x)",
     SpecialOp(x), None, CUSTOM_FUNCS_SPECIAL),
    ("WolframSpecial[x, a]", ['WolframSpecial', 'x', 'a'],
     "my_module.SpecialOp(x, Symbol('a'))", "my_module.SpecialOp(x, Symbol('a'))",
     SpecialOp(x, a), SYMS, CUSTOM_FUNCS_SPECIAL),
    ("Foo[x]", ['Foo', 'x'], "DirectFunc(x)", "DirectFunc(x)",
     DirectFunc(x), None, CUSTOM_FUNCS_DIRECT),
    ("Sin[FunctionCustom[x]]", ['Sin', ['FunctionCustom', 'x']],
     "sympy.sin(my_module.FunctionCustom(x))", "sympy.sin(my_module.FunctionCustom(x))",
     sin(FunctionCustom(x)), None, CUSTOM_FUNCS_MODULE),
    ("FunctionCustom[Sin[x]]", ['FunctionCustom', ['Sin', 'x']],
     "my_module.FunctionCustom(sympy.sin(x))", "my_module.FunctionCustom(sympy.sin(x))",
     FunctionCustom(sin(x)), None, CUSTOM_FUNCS_MODULE),
    ("FunctionCustom[x] + x", ['Plus', ['FunctionCustom', 'x'], 'x'],
     "(my_module.FunctionCustom(x) + x)", "(my_module.FunctionCustom(x) + x)",
     FunctionCustom(x) + x, None, CUSTOM_FUNCS_MODULE),
    ("FunctionCustom[x]^2", ['Power', ['FunctionCustom', 'x'], '2'],
     "(my_module.FunctionCustom(x))**(Integer(2))", "(my_module.FunctionCustom(x))**(Integer(2))",
     FunctionCustom(x)**2, None, CUSTOM_FUNCS_MODULE),
    ("FunctionCustom[x] + Foo[x]",
     ['Plus', ['FunctionCustom', 'x'], ['Foo', 'x']],
     "(my_module.FunctionCustom(x) + DirectFunc(x))",
     "(my_module.FunctionCustom(x) + DirectFunc(x))",
     FunctionCustom(x) + DirectFunc(x), None, CUSTOM_FUNCS_MULTI),
    ("FunctionCustom[a + b*x]",
     ['FunctionCustom', ['Plus', 'a', ['Times', 'b', 'x']]],
     "my_module.FunctionCustom((Symbol('a') + (Symbol('b') * x)))",
     "my_module.FunctionCustom((Symbol('a') + (Symbol('b') * x)))",
     FunctionCustom(a + b*x), SYMS, CUSTOM_FUNCS_MODULE),
]

CUSTOM_CASE_IDS = [c[0] for c in CUSTOM_CASES]


# ---------------------------------------------------------------------------
# Tests for custom_functions
# ---------------------------------------------------------------------------

@requires_mathematica_parser
class TestCustomFunctionsFFL:
    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_ffl_structure(self, mma_expr, expected_ffl, expected_code,
                           expected_short_code, expected_sympy,
                           extra_symbols, custom_functions):
        assert mathematica_to_ffl(mma_expr) == expected_ffl


@requires_mathematica_parser
class TestCustomFunctionsCode:
    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_code_string(self, mma_expr, expected_ffl, expected_code,
                         expected_short_code, expected_sympy,
                         extra_symbols, custom_functions):
        code, _defs, _symbols = mathematica_to_sympy_code(
            mma_expr, custom_functions=custom_functions, reserved_symbols=_X)
        assert code == expected_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_code_eval(self, mma_expr, expected_ffl, expected_code,
                       expected_short_code, expected_sympy,
                       extra_symbols, custom_functions):
        ns = {}
        code, _defs, _symbols = mathematica_to_sympy_code(
            mma_expr, custom_functions=custom_functions
        , namespace=ns, reserved_symbols=_X)
        if extra_symbols:
            ns.update(extra_symbols)
        result = eval(code, ns)
        assert_sympy_equal(result, expected_sympy)


@requires_mathematica_parser
class TestCustomFunctionsShortCode:
    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_short_code_string(self, mma_expr, expected_ffl, expected_code,
                               expected_short_code, expected_sympy,
                               extra_symbols, custom_functions):
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = mathematica_to_sympy_short_code(
            mma_expr, namespace=ns,
            custom_functions=custom_functions, reserved_symbols=_X,
        )
        assert short == expected_short_code

    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_short_code_eval(self, mma_expr, expected_ffl, expected_code,
                             expected_short_code, expected_sympy,
                             extra_symbols, custom_functions):
        ns = dict(extra_symbols or {})
        short, _defs, _symbols = mathematica_to_sympy_short_code(
            mma_expr, namespace=ns,
            custom_functions=custom_functions, reserved_symbols=_X,
        )
        result = eval(short, ns)
        assert_sympy_equal(result, expected_sympy)


@requires_mathematica_parser
class TestCustomFunctionsSympy:
    @pytest.mark.parametrize(
        "mma_expr,expected_ffl,expected_code,expected_short_code,"
        "expected_sympy,extra_symbols,custom_functions",
        CUSTOM_CASES, ids=CUSTOM_CASE_IDS,
    )
    def test_expression(self, mma_expr, expected_ffl, expected_code,
                        expected_short_code, expected_sympy,
                        extra_symbols, custom_functions):
        ns = dict(extra_symbols or {})
        result = mathematica_to_sympy(
            mma_expr, namespace=ns,
            custom_functions=custom_functions, reserved_symbols=_X,
        )
        assert_sympy_equal(result, expected_sympy)

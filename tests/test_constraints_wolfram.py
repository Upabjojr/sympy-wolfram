# -*- coding: utf-8 -*-
"""Tests for the standard-Wolfram constraint predicates in sympy_wolfram.

FreeQ, IntegerQ, PositiveQ and MemberQ are Mathematica standard-library
constraints (not Rubi-specific), so their :class:`MathematicaConstraint`
subclasses live in ``sympy_wolfram.constraints_wolfram``.  These tests moved here
from ``rubi_integrate/tests`` when the classes were relocated.
"""
import sympy
from sympy import Symbol, Integer, Rational, Float, pi, sin
from sympy.logic.boolalg import Boolean, Not

from sympy_matching.wild import WildSymbol
from sympy_wolfram.objects import MathematicaExpr
from sympy_wolfram.constraints import MathematicaConstraint
from sympy_wolfram.constraints_wolfram import FreeQ, IntegerQ, PositiveQ, NumberQ, AtomQ, PolynomialQ


x = Symbol('x')


# ---------------------------------------------------------------------------
# Base-class behaviour (Boolean / MathematicaExpr) via FreeQ
# ---------------------------------------------------------------------------

class TestMathematicaConstraintBoolean:
    """Verify MathematicaConstraint inherits from SymPy Boolean and MathematicaExpr."""

    def test_inheritance(self):
        assert issubclass(MathematicaConstraint, Boolean)

    def test_inherits_mathematica_expr(self):
        assert issubclass(MathematicaConstraint, MathematicaExpr)
        assert isinstance(FreeQ('a', x), MathematicaExpr)

    def test_instance_is_boolean(self):
        fq = FreeQ('a', x)
        assert isinstance(fq, Boolean)

    def test_not_composition(self):
        fq = FreeQ('a', x)
        neg = Not(fq)
        assert isinstance(neg, Boolean)


# ---------------------------------------------------------------------------
# FreeQ
# ---------------------------------------------------------------------------

class TestFreeQ:
    """Tests for FreeQ constraint."""

    def test_free_of_x(self):
        c = FreeQ('a', x)
        assert c.check(a=Integer(5)) == True
        assert c.check(a=Symbol('y')) == True
        assert c.check(a=x) == False
        assert c.check(a=x + 1) == False

    def test_free_of_x_expression(self):
        c = FreeQ('a', x)
        assert c.check(a=sin(Symbol('y'))) == True
        assert c.check(a=sin(x)) == False

    def test_repr(self):
        c = FreeQ('a', x)
        assert 'FreeQ' in repr(c)


class TestFreeQListForm:
    """Tests for FreeQ accepting list of variable names."""

    def test_single_variable(self):
        # variables auto-computed from WildSymbol instances
        a_ = WildSymbol('a')
        fq = FreeQ(a_, x)
        assert fq.variables == ('a',)
        assert fq.check(a=Symbol('a')) == True
        assert fq.check(a=x + 1) == False

    def test_list_all_free(self):
        a_, b_, c_ = WildSymbol('a'), WildSymbol('b'), WildSymbol('c')
        fq = FreeQ([a_, b_, c_], x)
        assert fq.variables == ('a', 'b', 'c')
        assert fq.check(a=Symbol('a'), b=Symbol('b'), c=Symbol('c')) == True

    def test_list_one_not_free(self):
        fq = FreeQ(['a', 'b', 'c'], x)
        assert fq.check(a=Symbol('a'), b=x + 1, c=Symbol('c')) == False

    def test_list_all_not_free(self):
        fq = FreeQ(['a', 'b'], x)
        assert fq.check(a=x, b=x**2) == False

    def test_tuple_form(self):
        a_, b_ = WildSymbol('a'), WildSymbol('b')
        fq = FreeQ((a_, b_), x)
        assert fq.variables == ('a', 'b')
        assert fq.check(a=Integer(1), b=Integer(2)) == True

    def test_repr_single(self):
        fq = FreeQ('a', x)
        assert 'FreeQ' in repr(fq)
        assert 'a' in repr(fq)

    def test_repr_list(self):
        fq = FreeQ(['a', 'b'], x)
        r = repr(fq)
        assert 'FreeQ' in r
        assert 'a' in r and 'b' in r


# ---------------------------------------------------------------------------
# IntegerQ
# ---------------------------------------------------------------------------

class TestIntegerQ:
    """Tests for IntegerQ constraint."""

    def test_integer(self):
        c = IntegerQ('n')
        assert c.check(n=Integer(5)) == True
        assert c.check(n=Integer(-3)) == True
        assert c.check(n=Integer(0)) == True

    def test_not_integer(self):
        c = IntegerQ('n')
        assert c.check(n=Rational(1, 2)) == False
        assert c.check(n=Float(3.14)) == False
        assert c.check(n=x) == False


# ---------------------------------------------------------------------------
# PositiveQ
# ---------------------------------------------------------------------------

class TestPositiveQ:
    """Tests for PositiveQ constraint."""

    def test_positive(self):
        c = PositiveQ('n')
        assert c.check(n=Integer(5)) == True
        assert c.check(n=Rational(1, 2)) == True
        assert c.check(n=pi) == True

    def test_not_positive(self):
        c = PositiveQ('n')
        assert c.check(n=Integer(-5)) == False
        assert c.check(n=Integer(0)) == False


# ---------------------------------------------------------------------------
# NumberQ / AtomQ / PolynomialQ (standard Wolfram predicates lifted here)
# ---------------------------------------------------------------------------

class TestNumberQ:
    """Tests for NumberQ constraint (explicit numbers only)."""

    def test_explicit_numbers(self):
        c = NumberQ('n')
        assert c.check(n=Integer(2)) is True
        assert c.check(n=Rational(3, 2)) is True
        assert c.check(n=Integer(2) + 3 * sympy.I) is True

    def test_not_numbers(self):
        c = NumberQ('n')
        assert c.check(n=pi) is False
        assert c.check(n=sympy.sqrt(2)) is False
        assert c.check(n=x) is False


class TestAtomQ:
    """Tests for AtomQ constraint."""

    def test_atoms(self):
        c = AtomQ('a')
        assert c.check(a=x) is True
        assert c.check(a=Integer(5)) is True

    def test_not_atoms(self):
        c = AtomQ('a')
        assert c.check(a=x + 1) is False


class TestPolynomialQ:
    """Tests for PolynomialQ constraint (polynomial in the integration variable)."""

    def test_polynomial(self):
        c = PolynomialQ('u', x)
        assert c.check(u=x ** 3) is True
        assert c.check(u=Integer(1) + x + x ** 2) is True

    def test_not_polynomial(self):
        c = PolynomialQ('u', x)
        assert c.check(u=sympy.sqrt(x)) is False

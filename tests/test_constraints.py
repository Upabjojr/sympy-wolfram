# -*- coding: utf-8 -*-
"""Tests for sympy_wolfram.constraints.MathematicaConstraint base class.

(Formerly sympy_matching.constraints.RubiConstraint — renamed and moved up into
sympy_wolfram so it can inherit from MathematicaExpr.)

Covers (without importing rubi_integrate):
- Module location: MathematicaConstraint lives in sympy_wolfram, NOT rubi_integrate
- No import of rubi_integrate anywhere in sympy_wolfram.constraints
- Dual inheritance: MathematicaExpr AND Boolean, plus logic composition
- Argument normalisation: str->Symbol, int->Integer, list->tuple, dict->tuple-of-pairs
- The SymPy invariant: constraint == constraint.func(*constraint.args)
- Hash consistency (equal objects have equal hashes)
- JSON round-trip via sympy_matching.json_ext
"""
import sys
import os
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import sympy
from sympy import Symbol, Integer, Rational, Tuple
from sympy.logic.boolalg import Boolean, Not, And, Or

import sympy_matching  # registers json_ext + conversion handlers
from sympy_wolfram.constraints import MathematicaConstraint
from sympy_wolfram.objects import MathematicaExpr
from sympy_matching.json_ext import serialize_wrapped_value, deserialize_wrapped_value


# ---------------------------------------------------------------------------
# Local test subclass (avoids rubi_integrate dependency)
# ---------------------------------------------------------------------------

class _TestPredicate(MathematicaConstraint):
    """A minimal MathematicaConstraint subclass for testing base-class features."""
    variables = ('x',)

    def check(self, **kwargs):
        # True if the value bound to 'x' is an integer
        val = kwargs.get('x')
        if val is None:
            return False
        return val.is_integer is True


class _TestBinaryPredicate(MathematicaConstraint):
    """A two-argument MathematicaConstraint subclass for testing."""
    variables = ('a', 'b')

    def check(self, **kwargs):
        a = kwargs.get('a')
        b = kwargs.get('b')
        if a is None or b is None:
            return False
        return a == b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip(constraint):
    """Serialize then deserialize a constraint; return the restored object."""
    data = serialize_wrapped_value(constraint)
    return deserialize_wrapped_value(data)


# ---------------------------------------------------------------------------
# 1. Module location: sympy_wolfram has NO rubi_integrate import
# ---------------------------------------------------------------------------

class TestModuleLocation:
    """MathematicaConstraint belongs to sympy_wolfram, not rubi_integrate."""

    def test_module_is_sympy_wolfram(self):
        assert MathematicaConstraint.__module__ == 'sympy_wolfram.constraints'

    def test_no_rubi_import_in_module(self):
        """sympy_wolfram.constraints must not import anything from rubi_integrate."""
        import importlib
        mod = importlib.import_module('sympy_wolfram.constraints')
        tree = ast.parse(open(mod.__file__).read())
        rubi_imports = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any(
                'rubi' in getattr(a, 'name', '') or
                'rubi' in (getattr(node, 'module', '') or '')
                for a in getattr(node, 'names', [node])
            )
        ]
        assert not rubi_imports, f"Found rubi_integrate imports: {rubi_imports}"


# ---------------------------------------------------------------------------
# 2. Dual inheritance: MathematicaExpr AND Boolean
# ---------------------------------------------------------------------------

class TestDualInheritance:
    def test_is_boolean_subclass(self):
        assert issubclass(_TestPredicate, Boolean)
        assert issubclass(_TestPredicate, MathematicaConstraint)

    def test_is_mathematica_expr_subclass(self):
        assert issubclass(MathematicaConstraint, MathematicaExpr)
        assert issubclass(_TestPredicate, MathematicaExpr)

    def test_instance_is_boolean_and_mathematica_expr(self):
        c = _TestPredicate('n')
        assert isinstance(c, Boolean)
        assert isinstance(c, MathematicaExpr)

    def test_doit_returns_self(self):
        # A constraint is a predicate, not a reducible expression.
        c = _TestPredicate('n')
        assert c.doit() is c

    def test_not_composition(self):
        c = _TestPredicate('n')
        result = Not(c)
        assert isinstance(result, Boolean)

    def test_and_composition(self):
        c1 = _TestPredicate('a')
        c2 = _TestPredicate('b')
        result = And(c1, c2)
        assert isinstance(result, Boolean)

    def test_or_composition(self):
        c1 = _TestPredicate('a')
        c2 = _TestPredicate('b')
        result = Or(c1, c2)
        assert isinstance(result, Boolean)


# ---------------------------------------------------------------------------
# 3. Argument normalisation
# ---------------------------------------------------------------------------

class TestArgNormalisation:
    """_normalize_constraint_arg converts Python primitives to SymPy."""

    def test_str_to_symbol(self):
        c = _TestPredicate('n_')   # trailing _ stripped
        assert c.args[0] == Symbol('n')

    def test_str_no_underscore(self):
        c = _TestPredicate('n')
        assert c.args[0] == Symbol('n')

    def test_int_to_integer(self):
        c = _TestBinaryPredicate('n', 2)
        assert c.args[1] == Integer(2)

    def test_list_to_tuple(self):
        c = _TestPredicate(['a', 'b'])
        assert isinstance(c.args[0], Tuple)
        assert c.args[0] == Tuple(Symbol('a'), Symbol('b'))

    def test_sympy_passthrough(self):
        a = Symbol('a')
        c = _TestPredicate(a)
        assert c.args[0] is a


# ---------------------------------------------------------------------------
# 4. The SymPy invariant: constraint == constraint.func(*constraint.args)
# ---------------------------------------------------------------------------

class TestInvariant:
    """constraint == constraint.func(*constraint.args) for the test subclass."""

    @pytest.fixture(params=[
        _TestPredicate('n_'),
        _TestPredicate('a'),
        _TestPredicate(['a', 'b']),
        _TestBinaryPredicate('a', 'b'),
        _TestBinaryPredicate('x_', 2),
    ], ids=lambda c: type(c).__name__ + str(c.args))
    def constraint(self, request):
        return request.param

    def test_func_args_identity(self, constraint):
        reconstructed = constraint.func(*constraint.args)
        assert constraint == reconstructed

    def test_hash_consistency(self, constraint):
        reconstructed = constraint.func(*constraint.args)
        assert hash(constraint) == hash(reconstructed)

    def test_args_are_hashable(self, constraint):
        for a in constraint.args:
            hash(a)  # must not raise


# ---------------------------------------------------------------------------
# 5. JSON serialization round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundtrip:
    """Constraints must survive serialize/deserialize with equality."""

    @pytest.fixture(params=[
        _TestPredicate('n'),
        _TestBinaryPredicate('a', 'b'),
    ], ids=lambda c: type(c).__name__ + str(c.args))
    def constraint(self, request):
        return request.param

    def test_roundtrip(self, constraint):
        restored = _roundtrip(constraint)
        assert constraint == restored

    def test_roundtrip_preserves_check(self):
        c = _TestPredicate('x')
        restored = _roundtrip(c)
        assert c.check(x=Integer(3)) == restored.check(x=Integer(3))
        assert c.check(x=Rational(1, 2)) == restored.check(x=Rational(1, 2))

# -*- coding: utf-8 -*-
"""Tests for sympy_wolfram.objects.

Test cases derived from Wolfram Mathematica documentation examples.
"""
import pytest
import sympy
from sympy import Integer, Rational, S, Symbol, sqrt

from sympy_wolfram.objects import (
    Block,
    Catch,
    CompoundExpression,
    Do,
    Head,
    If,
    List,
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
import sympy_wolfram.objects as _me
from sympy_wolfram.interpreter import mathematica_to_sympy
from sympy_wolfram.parser import parse_mathematica_to_fullformlist

# Round-trip tests parse Mathematica SOURCE STRINGS, which needs sympy's
# parse_mathematica_to_fullformlist — absent from released sympy (<= 1.14).
requires_mathematica_parser = pytest.mark.skipif(
    parse_mathematica_to_fullformlist is None,
    reason='sympy.parsing.mathematica.parse_mathematica_to_fullformlist not '
           'available in this sympy version (needs sympy > 1.14)')

# ---------------------------------------------------------------------------
# Custom-functions mapping for round-trip tests
# ---------------------------------------------------------------------------
# Passed as custom_functions= to mathematica_to_sympy() so that the parser
# produces the same unevaluated SymPy objects as direct Python constructors.
# Each entry: WolframHead -> ('me.ClassName', module_object).  The converter
# emits 'me.ClassName(args...)' and registers the module as 'me' in the eval
# namespace.
_MATH_EXPR_FUNCS = {
    'List':               ('me.List',               _me),
    'Set':                ('me.Set',                _me),
    'If':                 ('me.If',                 _me),
    'With':               ('me.With',               _me),
    'Module':             ('me.Module',             _me),
    'Block':              ('me.Block',              _me),
    'CompoundExpression': ('me.CompoundExpression', _me),
    'Return':             ('me.Return',             _me),
    'Do':                 ('me.Do',                 _me),
    'Scan':               ('me.Scan',               _me),
    'Throw':              ('me.Throw',              _me),
    'Catch':              ('me.Catch',              _me),
    'Sow':                ('me.Sow',                _me),
    'Reap':               ('me.Reap',               _me),
    'Head':               ('me.Head',               _me),
}


class TestList:
    """Tests for List: Mathematica List container."""

    def test_list_construction(self):
        """List[1, 2, 3]"""
        lst = List(1, 2, 3)
        assert len(lst.args) == 3
        assert lst.args == (Integer(1), Integer(2), Integer(3))

    def test_list_iteration(self):
        """List items can be iterated."""
        lst = List(1, 2, 3)
        assert list(lst) == [Integer(1), Integer(2), Integer(3)]

    def test_list_doit_returns_self(self):
        """List.doit() returns the list unchanged."""
        lst = List(1, 2, 3)
        assert lst.doit() == lst


class TestSet:
    """Tests for Set: Mathematica variable binding marker."""

    def test_set_construction(self):
        """Set[x, 5]"""
        x = Symbol('x')
        binding = Set(x, Integer(5))
        assert binding.args == (x, Integer(5))

    def test_set_doit_returns_self(self):
        """Set is a structural marker, not evaluated."""
        x = Symbol('x')
        binding = Set(x, Integer(5))
        assert binding.doit() == binding


class TestIf:
    """Tests for If: Mathematica conditional expression.

    Documentation: https://reference.wolfram.com/language/ref/If.html
    """

    def test_if_true_branch(self):
        """If[True, 1, 0] -> 1"""
        result = If(S.true, Integer(1), Integer(0)).doit()
        assert result == Integer(1)

    def test_if_false_branch(self):
        """If[False, 1, 0] -> 0"""
        result = If(S.false, Integer(1), Integer(0)).doit()
        assert result == Integer(0)

    def test_if_two_args_true(self):
        """If[True, 42] -> 42"""
        result = If(S.true, Integer(42)).doit()
        assert result == Integer(42)

    def test_if_two_args_false(self):
        """If[False, 42] -> Null"""
        result = If(S.false, Integer(42)).doit()
        assert result == Null

    def test_if_unknown_returns_unevaluated(self):
        """If[x > 0, 1, -1] stays unevaluated when condition is symbolic."""
        x = Symbol('x')
        cond = sympy.Gt(x, 0)
        result = If(cond, Integer(1), Integer(-1)).doit()
        assert isinstance(result, If)

    def test_if_four_args_unknown(self):
        """If[x > 0, 1, -1, 0] -> 0 when condition is neither True nor False."""
        x = Symbol('x')
        cond = sympy.Gt(x, 0)
        result = If(cond, Integer(1), Integer(-1), Integer(0)).doit()
        assert result == Integer(0)

    def test_if_evaluates_a_matchq_constraint_condition(self):
        """A predicate used as an If CONDITION (e.g. If[MatchQ[f, f1*Complex(0,j)], ...]
        in a rule's replacement) is a MathematicaConstraint, not a raw boolean. Wolfram
        evaluates MatchQ[...] to True/False -- resolving its LOCAL pattern wildcards (f1/j)
        internally -- so the If picks a branch. If.doit must do the same via .check();
        otherwise the If stays unevaluated and the local wildcards leak into the result.
        Cross-checked on real Wolfram: MatchQ[b/d, f1*Complex[0,j]] is False, [2 I b] True.
        """
        from rubi_integrate.utils.constraints_wolfram import MatchQ
        from sympy_matching.wild import WildSymbol
        from sympy import I, symbols
        b, d = symbols('b d')
        f1, j = WildSymbol('f1'), WildSymbol('j')
        A, B = Symbol('A'), Symbol('B')
        # b/d is real -> does NOT match a pure-imaginary pattern -> else branch, no wildcards
        res_real = If(MatchQ(b / d, f1 * I * j), A, B).doit()
        assert res_real == B
        assert not any(getattr(s, 'wildcard_name', None) for s in res_real.free_symbols)
        # 2*I*b is imaginary -> matches -> then branch
        assert If(MatchQ(2 * I * b, f1 * I * j), A, B).doit() == A


class TestWith:
    """Tests for With: Mathematica local constant substitution.

    Documentation: https://reference.wolfram.com/language/ref/With.html
    """

    def test_with_single_binding(self):
        """With[{x = 5}, x + 1] -> 6"""
        x = Symbol('x')
        result = With(List(Set(x, Integer(5))), x + Integer(1)).doit()
        assert result == Integer(6)

    def test_with_multiple_bindings(self):
        """With[{x = 2, y = 3}, x * y] -> 6"""
        x, y = Symbol('x'), Symbol('y')
        bindings = List(Set(x, Integer(2)), Set(y, Integer(3)))
        result = With(bindings, x * y).doit()
        assert result == Integer(6)

    def test_with_nested_expression(self):
        """With[{a = 2}, a^2 + a + 1] -> 7"""
        a = Symbol('a')
        result = With(List(Set(a, Integer(2))), a**2 + a + Integer(1)).doit()
        assert result == Integer(7)

    def test_with_return_handling(self):
        """Return within With stops evaluation and returns the value."""
        x = Symbol('x')
        result = With(List(Set(x, Integer(10))), Return(x * Integer(2))).doit()
        assert result == Integer(20)


class TestModule:
    """Tests for Module: Mathematica lexical scoping with fresh symbols.

    Documentation: https://reference.wolfram.com/language/ref/Module.html
    """

    def test_module_single_local(self):
        """Module[{x = 5}, x + 1] -> 6"""
        x = Symbol('x')
        result = Module(List(Set(x, Integer(5))), x + Integer(1)).doit()
        assert result == Integer(6)

    def test_module_local_renamed(self):
        """Module introduces fresh symbols."""
        x = Symbol('x')
        result = Module(List(x), x).doit()
        assert isinstance(result, Symbol)
        assert result.name.startswith('x')

    def test_module_multiple_locals(self):
        """Module[{a = 2, b = 3}, a + b] -> 5"""
        a, b = Symbol('a'), Symbol('b')
        bindings = List(Set(a, Integer(2)), Set(b, Integer(3)))
        result = Module(bindings, a + b).doit()
        assert result == Integer(5)


class TestBlock:
    """Tests for Block: Mathematica dynamic scoping with temporary values.

    Documentation: https://reference.wolfram.com/language/ref/Block.html
    """

    def test_block_single_binding(self):
        """Block[{x = 10}, x + 5] -> 15"""
        x = Symbol('x')
        result = Block(List(Set(x, Integer(10))), x + Integer(5)).doit()
        assert result == Integer(15)

    def test_block_multiple_bindings(self):
        """Block[{a = 3, b = 7}, a * b] -> 21"""
        a, b = Symbol('a'), Symbol('b')
        bindings = List(Set(a, Integer(3)), Set(b, Integer(7)))
        result = Block(bindings, a * b).doit()
        assert result == Integer(21)


class TestCompoundExpression:
    """Tests for CompoundExpression: sequential evaluation.

    Documentation: https://reference.wolfram.com/language/ref/CompoundExpression.html
    """

    def test_compound_returns_last(self):
        """CompoundExpression[1, 2, 3] -> 3"""
        result = CompoundExpression(Integer(1), Integer(2), Integer(3)).doit()
        assert result == Integer(3)

    def test_compound_empty(self):
        """CompoundExpression[] -> Null"""
        result = CompoundExpression().doit()
        assert result == Null

    def test_compound_single_expr(self):
        """CompoundExpression[42] -> 42"""
        result = CompoundExpression(Integer(42)).doit()
        assert result == Integer(42)

    def test_scoping_constructs_accept_dict_bindings(self):
        """Module/With/Block accept a Python-dict binding form in addition to the
        Mathematica `List(Set(...))` tree -- the codegen now emits the dict form
        (`Module({r: v1, s: v2, k: None}, body)`). A None value is an UNINITIALISED
        local. Dict and List forms must be equivalent."""
        from sympy_wolfram.objects import Block
        r, s, k = Symbol('r'), Symbol('s'), Symbol('k')
        i = Integer
        # dict form works for all three constructs
        assert With({r: i(2), s: i(3)}, r**2 + s).doit() == Integer(7)
        assert Module({r: i(4)}, r*(r + i(1))).doit() == Integer(20)
        assert Block({r: i(2)}, r + i(1)).doit() == Integer(3)
        # None = uninitialised local, assigned later inside the body
        m = Module({r: i(2), k: None},
                   CompoundExpression(Set(k, i(3)*r), r + k))
        assert m.doit() == Integer(8)
        # dict form is equivalent to the explicit List(Set(...)) form
        assert (Module({r: i(2), s: i(3)}, r + s).doit()
                == Module(List(Set(r, i(2)), Set(s, i(3))), r + s).doit())

    def test_scoping_constructs_have_all_sympy_args(self):
        """After construction, EVERY node reachable through ``.args`` must be a
        ``sympy.Basic``. The dict/``None`` binding form is normalised through
        ``_bindings_list_from_dict`` + ``sympify``, so no raw Python ``dict``,
        ``None`` or ``int`` may leak into the expression tree (which would break
        ``xreplace``, ``srepr`` round-trips and matcher traversal)."""
        from sympy import Basic
        from sympy_wolfram.objects import Block

        def non_sympy_nodes(expr, path='root'):
            if not isinstance(expr, Basic):
                return [(path, type(expr).__name__, repr(expr)[:40])]
            bad = []
            for i, arg in enumerate(expr.args):
                bad += non_sympy_nodes(arg, f"{path}.args[{i}]")
            return bad

        r, s, k, u = (Symbol(n) for n in ('r', 's', 'k', 'u'))
        i = Integer
        cases = [
            # dict form with initialised + uninitialised (None) locals
            Module({r: i(2), s: i(3), k: None, u: None},
                   CompoundExpression(Set(u, r*k), r + u)),
            With({r: i(2)}, r + i(1)),
            Block({r: i(2), k: None}, r + k),
            # a plain Python int as a dict value must be sympified, not left raw
            Module({r: 5}, r),
            # the equivalent explicit List(Set(...)) tree form
            Module(List(Set(r, i(2)), k), r + k),
        ]
        for expr in cases:
            bad = non_sympy_nodes(expr)
            assert not bad, f"non-SymPy nodes in {type(expr).__name__}.args: {bad}"

    def test_compound_set_binds_for_later_statements(self):
        """CompoundExpression[Set[u, val], body] BINDS u for the following
        statements (Mathematica's assignment side effect). Verified vs Mathematica:
        ``(u = 5; u + 1)`` -> 6 and ``(u = 3; v = 2 u; v + u)`` -> 9. Rubi relies on
        this in Module[{..., k, u}, u = Int[f(k)]; ... Sum[u, {k, 1, N}]]; without it
        u (a scoping Dummy) leaked unresolved into the Sum and gave wrong integrals
        (e.g. sqrt(x)*(A+B*x**3)/(a+b*x**3))."""
        u, v = Symbol('u'), Symbol('v')
        assert CompoundExpression(Set(u, Integer(5)), u + Integer(1)).doit() == Integer(6)
        assert CompoundExpression(
            Set(u, Integer(3)), Set(v, Integer(2)*u), v + u).doit() == Integer(9)

    def test_compound_set_scoping_matches_mathematica(self):
        """Set inside a CompoundExpression is scoped to the ENCLOSING Module/With
        (via the Dummy renaming those constructs apply at construction), and does not
        leak into nested scopes. Every expected value here was checked against real
        Mathematica (`<<Rubi`; wolframscript`), including the Rubi ``u = f(k); Sum[u,
        {k,1,N}]`` pattern and nested/With shadowing.
        """
        from sympy_wolfram.objects import Module, With, List
        from rubi_integrate.utils.rubi_utils import Sum
        u, v, k = Symbol('u'), Symbol('v'), Symbol('k')
        i = Integer
        cases = [
            (Module(List(u), CompoundExpression(Set(u, i(5)), u + i(1))), 6),
            # Rubi pattern: u bound to an expression in the Sum index k
            (Module(List(k, u), CompoundExpression(Set(u, k**2), Sum(u, List(k, i(1), i(3))))), 14),
            (Module(List(k, u), CompoundExpression(Set(u, k + i(10)), Sum(u, List(k, i(1), i(3))))), 36),
            # nested Module shadow: inner u=10 must NOT be clobbered by outer u=5
            (Module(List(u), CompoundExpression(Set(u, i(5)),
                Module(List(u), CompoundExpression(Set(u, i(10)), u)) + u)), 15),
            # With shadow
            (Module(List(u), CompoundExpression(Set(u, i(5)),
                With(List(Set(u, i(10))), u) + u)), 15),
            # reassignment sees the previous value
            (Module(List(u), CompoundExpression(Set(u, i(3)), Set(u, u + i(100)), u)), 103),
            (Module(List(u, v), CompoundExpression(Set(u, i(3)), Set(v, i(2)*u), v + u)), 9),
        ]
        for expr, want in cases:
            assert expr.doit() == Integer(want), (expr, expr.doit(), want)

    def test_setdelayed_holds_rhs_unlike_set(self):
        """SetDelayed (:=) HOLDS its RHS and re-evaluates it at use time; Set (=)
        fixes the value at assignment. The distinguishing case is a RHS variable
        reassigned AFTER the binding -- every expected value verified against real
        Mathematica (`<<Rubi`; wolframscript`):

            Module[{u,y}, y=2; u:=y^2; y=3; u]  -> 9  (SetDelayed re-evaluates y^2)
            Module[{u,y}, y=2; u =y^2; y=3; u]  -> 4  (Set fixed y^2 at y=2)
        """
        u, y = Symbol('u'), Symbol('y')
        i = Integer
        # basic: SetDelayed binds like Set when nothing is reassigned
        assert Module(List(u), CompoundExpression(
            SetDelayed(u, i(5)), u + i(1))).doit() == Integer(6)
        # the distinguishing behaviour
        assert Module(List(u, y), CompoundExpression(
            Set(y, i(2)), SetDelayed(u, y**2), Set(y, i(3)), u)).doit() == Integer(9)
        assert Module(List(u, y), CompoundExpression(
            Set(y, i(2)), Set(u, y**2), Set(y, i(3)), u)).doit() == Integer(4)
        # a self-referential delayed binding (Mathematica: unterminating recursion)
        # must TERMINATE here rather than hang -- the resolution is bounded.
        Module(List(u), CompoundExpression(
            SetDelayed(u, i(3)), SetDelayed(u, u + i(100)), u)).doit()


class TestReturn:
    """Tests for Return: Mathematica return from procedural constructs.

    Documentation: https://reference.wolfram.com/language/ref/Return.html
    """

    def test_return_unwrapped(self):
        """Return[5] within With[{}, Return[5]] -> 5"""
        result = With(List(), Return(Integer(5))).doit()
        assert result == Integer(5)

    def test_return_default_value(self):
        """Return[] -> Null"""
        result = With(List(), Return()).doit()
        assert result == Null


class TestDo:
    """Tests for Do: Mathematica iteration construct.

    Documentation: https://reference.wolfram.com/language/ref/Do.html
    """

    def test_do_single_iteration(self):
        """Do[expr, {5}] executes 5 times, returns Null."""
        result = Do(Integer(42), List(Integer(5))).doit()
        assert result == Null

    def test_do_with_iterator(self):
        """Do with i from 1 to 3."""
        i = Symbol('i')
        result = Do(Sow(i), List(i, Integer(3))).doit()
        assert result == Null

    def test_do_with_range(self):
        """Do[Sow[i], {i, 2, 4}] sows 2, 3, 4."""
        i = Symbol('i')
        reap_result = Reap(Do(Sow(i), List(i, Integer(2), Integer(4)))).doit()
        sown = reap_result.args[1]
        values = list(sown.args[0].args)
        assert values == [Integer(2), Integer(3), Integer(4)]

    def test_do_with_step(self):
        """Do[Sow[i], {i, 1, 5, 2}] sows 1, 3, 5."""
        i = Symbol('i')
        reap_result = Reap(Do(Sow(i), List(i, Integer(1), Integer(5), Integer(2)))).doit()
        values = list(reap_result.args[1].args[0].args)
        assert values == [Integer(1), Integer(3), Integer(5)]


class TestScan:
    """Tests for Scan: apply function for side effects.

    Documentation: https://reference.wolfram.com/language/ref/Scan.html
    """

    def test_scan_returns_null(self):
        """Scan[f, {1, 2, 3}] returns Null."""
        f = Symbol('f')
        result = Scan(f, List(Integer(1), Integer(2), Integer(3))).doit()
        assert result == Null

    def test_scan_with_sow(self):
        """Scan[Sow, {1, 2}] inside Reap collects values."""
        lst = List(Integer(1), Integer(2))
        reap_result = Reap(Scan(sympy.Lambda(Symbol('x'), Sow(Symbol('x'))), lst)).doit()
        values = list(reap_result.args[1].args[0].args)
        assert values == [Integer(1), Integer(2)]


class TestThrowCatch:
    """Tests for Throw and Catch: Mathematica exception mechanism.

    Documentation:
    - https://reference.wolfram.com/language/ref/Throw.html
    - https://reference.wolfram.com/language/ref/Catch.html
    """

    def test_catch_simple(self):
        """Catch[Throw[42]] -> 42"""
        result = Catch(Throw(Integer(42))).doit()
        assert result == Integer(42)

    def test_catch_nested_expression(self):
        """Catch[1 + Throw[5]] -> 5"""
        result = Catch(Integer(1) + Throw(Integer(5))).doit()
        assert result == Integer(5)

    def test_catch_with_tag(self):
        """Catch[Throw[val, tag], tag] -> val"""
        tag = Symbol('myTag')
        result = Catch(Throw(Integer(99), tag), tag).doit()
        assert result == Integer(99)

    def test_catch_mismatched_tag_propagates(self):
        """Catch with mismatched tag re-raises."""
        tag1, tag2 = Symbol('tag1'), Symbol('tag2')
        with pytest.raises(Exception):
            Catch(Throw(Integer(1), tag1), tag2).doit()

    def test_catch_no_throw(self):
        """Catch[5 + 3] -> 8 (no Throw means normal evaluation)."""
        result = Catch(Integer(5) + Integer(3)).doit()
        assert result == Integer(8)


class TestSowReap:
    """Tests for Sow and Reap: Mathematica value collection mechanism.

    Documentation:
    - https://reference.wolfram.com/language/ref/Sow.html
    - https://reference.wolfram.com/language/ref/Reap.html
    """

    def test_reap_simple(self):
        """Reap[Sow[1]; Sow[2]] -> {Null, {{1, 2}}}"""
        result = Reap(CompoundExpression(Sow(Integer(1)), Sow(Integer(2)))).doit()
        assert result.args[0] == Integer(2)
        inner = result.args[1].args[0]
        assert list(inner.args) == [Integer(1), Integer(2)]

    def test_sow_returns_value(self):
        """Sow[x] returns x."""
        x = Symbol('x')
        result = Sow(x).doit()
        assert result == x

    def test_reap_with_tag(self):
        """Reap with tag filter."""
        tag = Symbol('myTag')
        result = Reap(
            CompoundExpression(Sow(Integer(1), tag), Sow(Integer(2), tag)),
            tag
        ).doit()
        inner = result.args[1].args[0]
        assert list(inner.args) == [Integer(1), Integer(2)]

    def test_sow_outside_reap_does_nothing(self):
        """Sow outside Reap just returns the value."""
        result = Sow(Integer(42)).doit()
        assert result == Integer(42)


class TestHead:
    """Tests for Head: Mathematica expression head.

    Documentation: https://reference.wolfram.com/language/ref/Head.html
    """

    def test_head_integer(self):
        """Head[5] -> Integer"""
        result = Head(Integer(5)).doit()
        assert result == Symbol('Integer')

    def test_head_rational(self):
        """Head[2/3] -> Rational"""
        result = Head(Rational(2, 3)).doit()
        assert result == Symbol('Rational')

    def test_head_symbol(self):
        """Head[x] -> Symbol"""
        x = Symbol('x')
        result = Head(x).doit()
        assert result == Symbol('Symbol')

    def test_head_list(self):
        """Head[{1, 2}] -> List"""
        result = Head(List(Integer(1), Integer(2))).doit()
        assert result == Symbol('List')

    def test_head_add(self):
        """Head[a + b] -> Plus"""
        a, b = Symbol('a'), Symbol('b')
        result = Head(a + b).doit()
        assert result == Symbol('Plus')

    def test_head_mul(self):
        """Head[a * b] -> Times"""
        a, b = Symbol('a'), Symbol('b')
        result = Head(a * b).doit()
        assert result == Symbol('Times')

    def test_head_pow(self):
        """Head[a^b] -> Power"""
        a, b = Symbol('a'), Symbol('b')
        result = Head(a**b).doit()
        assert result == Symbol('Power')

    def test_head_function(self):
        """Head[Sin[x]] -> sin"""
        x = Symbol('x')
        result = Head(sympy.sin(x)).doit()
        assert result == Symbol('sin')


class TestNestedConstructs:
    """Tests for combinations of constructs."""

    def test_nested_with(self):
        """With[{x = 2}, With[{y = 3}, x + y]] -> 5"""
        x, y = Symbol('x'), Symbol('y')
        inner = With(List(Set(y, Integer(3))), x + y)
        outer = With(List(Set(x, Integer(2))), inner)
        assert outer.doit() == Integer(5)

    def test_module_in_do(self):
        """Do with Module scope."""
        i, j = Symbol('i'), Symbol('j')
        body = Module(List(Set(j, i * Integer(10))), Sow(j))
        result = Reap(Do(body, List(i, Integer(1), Integer(3)))).doit()
        values = list(result.args[1].args[0].args)
        assert values == [Integer(10), Integer(20), Integer(30)]

    def test_if_in_with(self):
        """With[{x = 5}, If[x > 3, 100, 0]]"""
        x = Symbol('x')
        body = If(sympy.Gt(x, Integer(3)), Integer(100), Integer(0))
        result = With(List(Set(x, Integer(5))), body).doit()
        assert result == Integer(100)


# ---------------------------------------------------------------------------
# Round-trip tests: Mathematica string → mathematica_to_sympy → expression
# ---------------------------------------------------------------------------


def assert_same_scope(result, expected):
    """Compare two scoping nodes without relying on binder identity.

    ``With``/``Module``/``Block`` bind their locals to fresh ``Dummy`` symbols, so
    two independently-constructed but alpha-equivalent nodes are deliberately NOT
    ``==`` (that is what stops an outside substitution capturing a local). Compare
    the construct and the value it evaluates to instead.
    """
    assert type(result) is type(expected), (type(result), type(expected))
    assert len(result.args[0].args) == len(expected.args[0].args)
    assert result.doit() == expected.doit()


@requires_mathematica_parser
class TestMathematicaToSympyRoundTrip:
    """Verify mathematica_to_sympy() builds the same unevaluated SymPy objects
    as direct Python constructors when given _MATH_EXPR_FUNCS as the remapping.

    Each test mirrors a case from one of the TestXxx classes above, using an
    explicit Mathematica function-call string (e.g. ``List[1,2,3]``,
    ``Set[x,5]``, ``With[List[Set[x,5]], Plus[x,1]]``) to verify the full
    parse → FFL → eval pipeline.  No ``.doit()`` is called; only the
    unevaluated object structure is checked.
    """

    # -- List -----------------------------------------------------------------

    def test_list_from_string(self):
        """List[1, 2, 3] parses to List(Integer(1), Integer(2), Integer(3))."""
        result = mathematica_to_sympy("List[1, 2, 3]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == List(Integer(1), Integer(2), Integer(3))

    # -- Set ------------------------------------------------------------------

    def test_set_from_string(self):
        """Set[x, 5] parses to Set(Symbol('x'), Integer(5))."""
        x = Symbol('x')
        result = mathematica_to_sympy("Set[x, 5]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Set(x, Integer(5))

    # -- If -------------------------------------------------------------------

    def test_if_true_branch_from_string(self):
        """If[True, 1, 0] parses to If(S.true, Integer(1), Integer(0))."""
        result = mathematica_to_sympy("If[True, 1, 0]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == If(S.true, Integer(1), Integer(0))

    def test_if_false_branch_from_string(self):
        """If[False, 1, 0] parses to If(S.false, Integer(1), Integer(0))."""
        result = mathematica_to_sympy("If[False, 1, 0]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == If(S.false, Integer(1), Integer(0))

    def test_if_two_args_from_string(self):
        """If[True, 42] parses to the 2-arg If form."""
        result = mathematica_to_sympy("If[True, 42]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == If(S.true, Integer(42))

    def test_if_four_args_from_string(self):
        """If[Greater[x, 0], 1, 0, 99] parses to the 4-arg If form."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "If[Greater[x, 0], 1, 0, 99]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == If(sympy.Gt(x, Integer(0)), Integer(1), Integer(0), Integer(99))

    # -- With -----------------------------------------------------------------

    def test_with_single_binding_from_string(self):
        """With[List[Set[x, 5]], Plus[x, 1]] parses correctly."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "With[List[Set[x, 5]], Plus[x, 1]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, With(List(Set(x, Integer(5))), x + Integer(1)))

    def test_with_multiple_bindings_from_string(self):
        """With[List[Set[x, 2], Set[y, 3]], Times[x, y]] parses correctly."""
        x, y = Symbol('x'), Symbol('y')
        result = mathematica_to_sympy(
            "With[List[Set[x, 2], Set[y, 3]], Times[x, y]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, With(List(Set(x, Integer(2)), Set(y, Integer(3))), x * y))

    def test_with_nested_expr_from_string(self):
        """With[List[Set[a, 2]], Times[a, Plus[a, 1]]] parses correctly."""
        a = Symbol('a')
        result = mathematica_to_sympy(
            "With[List[Set[a, 2]], Times[a, Plus[a, 1]]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, With(List(Set(a, Integer(2))), a * (a + Integer(1))))

    def test_with_return_from_string(self):
        """With[List[Set[x, 10]], Return[Times[x, 2]]] parses correctly."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "With[List[Set[x, 10]], Return[Times[x, 2]]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, With(List(Set(x, Integer(10))), Return(x * Integer(2))))

    # -- Module ---------------------------------------------------------------

    def test_module_with_init_from_string(self):
        """Module[List[Set[x, 5]], Plus[x, 1]] parses correctly."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "Module[List[Set[x, 5]], Plus[x, 1]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, Module(List(Set(x, Integer(5))), x + Integer(1)))

    def test_module_bare_local_from_string(self):
        """Module[List[x], x] parses correctly (uninitialized local).

        Not compared with `assert_same_scope`: an UNINITIALISED Module local
        evaluates to its fresh symbol, so two such Modules never evaluate equal --
        which is exactly Mathematica's behaviour (``Module[{x}, x]`` yields a new
        ``x$nnn`` each time).
        """
        x = Symbol('x')
        result = mathematica_to_sympy(
            "Module[List[x], x]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert isinstance(result, Module)
        assert len(result.args[0].args) == 1
        evaluated = result.doit()
        assert isinstance(evaluated, Dummy) and evaluated != x

    def test_module_multiple_locals_from_string(self):
        """Module[List[Set[a, 2], Set[b, 3]], Plus[a, b]] parses correctly."""
        a, b = Symbol('a'), Symbol('b')
        result = mathematica_to_sympy(
            "Module[List[Set[a, 2], Set[b, 3]], Plus[a, b]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, Module(List(Set(a, Integer(2)), Set(b, Integer(3))), a + b))

    # -- Block ----------------------------------------------------------------

    def test_block_single_binding_from_string(self):
        """Block[List[Set[x, 10]], Plus[x, 5]] parses correctly."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "Block[List[Set[x, 10]], Plus[x, 5]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, Block(List(Set(x, Integer(10))), x + Integer(5)))

    def test_block_multiple_bindings_from_string(self):
        """Block[List[Set[a, 3], Set[b, 7]], Times[a, b]] parses correctly."""
        a, b = Symbol('a'), Symbol('b')
        result = mathematica_to_sympy(
            "Block[List[Set[a, 3], Set[b, 7]], Times[a, b]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert_same_scope(result, Block(List(Set(a, Integer(3)), Set(b, Integer(7))), a * b))

    # -- CompoundExpression ---------------------------------------------------

    def test_compound_expression_from_string(self):
        """CompoundExpression[1, 2, 3] parses correctly."""
        result = mathematica_to_sympy(
            "CompoundExpression[1, 2, 3]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == CompoundExpression(Integer(1), Integer(2), Integer(3))

    def test_compound_single_expr_from_string(self):
        """CompoundExpression[42] parses correctly."""
        result = mathematica_to_sympy(
            "CompoundExpression[42]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == CompoundExpression(Integer(42))

    # -- Return ---------------------------------------------------------------

    def test_return_value_from_string(self):
        """Return[5] parses to Return(Integer(5))."""
        result = mathematica_to_sympy("Return[5]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Return(Integer(5))

    def test_return_null_from_string(self):
        """Return[Null] parses to Return(Null) (the Null sentinel)."""
        result = mathematica_to_sympy("Return[Null]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Return(Null)

    # -- Do -------------------------------------------------------------------

    def test_do_count_from_string(self):
        """Do[42, List[5]] parses to the fixed-count Do form."""
        result = mathematica_to_sympy(
            "Do[42, List[5]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Do(Integer(42), List(Integer(5)))

    def test_do_iterator_from_string(self):
        """Do[Sow[i], List[i, 3]] parses correctly (implicit start=1)."""
        i = Symbol('i')
        result = mathematica_to_sympy(
            "Do[Sow[i], List[i, 3]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Do(Sow(i), List(i, Integer(3)))

    def test_do_with_range_from_string(self):
        """Do[Sow[i], List[i, 2, 4]] parses correctly."""
        i = Symbol('i')
        result = mathematica_to_sympy(
            "Do[Sow[i], List[i, 2, 4]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Do(Sow(i), List(i, Integer(2), Integer(4)))

    def test_do_with_step_from_string(self):
        """Do[Sow[i], List[i, 1, 5, 2]] parses correctly (step form)."""
        i = Symbol('i')
        result = mathematica_to_sympy(
            "Do[Sow[i], List[i, 1, 5, 2]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Do(Sow(i), List(i, Integer(1), Integer(5), Integer(2)))

    # -- Scan -----------------------------------------------------------------

    def test_scan_from_string(self):
        """Scan[f, List[1, 2, 3]] parses to Scan(Symbol('f'), List(...))."""
        f = Symbol('f')
        result = mathematica_to_sympy(
            "Scan[f, List[1, 2, 3]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Scan(f, List(Integer(1), Integer(2), Integer(3)))

    # -- Throw / Catch --------------------------------------------------------

    def test_throw_from_string(self):
        """Throw[42] parses to Throw(Integer(42))."""
        result = mathematica_to_sympy("Throw[42]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Throw(Integer(42))

    def test_throw_with_tag_from_string(self):
        """Throw[99, myTag] parses to Throw(Integer(99), Symbol('myTag'))."""
        tag = Symbol('myTag')
        result = mathematica_to_sympy(
            "Throw[99, myTag]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Throw(Integer(99), tag)

    def test_catch_simple_from_string(self):
        """Catch[Throw[42]] parses to Catch(Throw(Integer(42)))."""
        result = mathematica_to_sympy(
            "Catch[Throw[42]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Catch(Throw(Integer(42)))

    def test_catch_with_tag_from_string(self):
        """Catch[Throw[99, myTag], myTag] parses to the tagged Catch form."""
        tag = Symbol('myTag')
        result = mathematica_to_sympy(
            "Catch[Throw[99, myTag], myTag]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Catch(Throw(Integer(99), tag), tag)

    def test_catch_no_throw_from_string(self):
        """Catch[Plus[5, 3]] parses to Catch(Integer(8)) (arithmetic evaluated)."""
        result = mathematica_to_sympy(
            "Catch[Plus[5, 3]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Catch(Integer(8))

    # -- Sow / Reap -----------------------------------------------------------

    def test_sow_from_string(self):
        """Sow[42] parses to Sow(Integer(42))."""
        result = mathematica_to_sympy("Sow[42]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Sow(Integer(42))

    def test_reap_sow_from_string(self):
        """Reap[Sow[1]] parses to Reap(Sow(Integer(1)))."""
        result = mathematica_to_sympy(
            "Reap[Sow[1]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Reap(Sow(Integer(1)))

    def test_reap_compound_from_string(self):
        """Reap[CompoundExpression[Sow[1], Sow[2]]] parses correctly."""
        result = mathematica_to_sympy(
            "Reap[CompoundExpression[Sow[1], Sow[2]]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Reap(CompoundExpression(Sow(Integer(1)), Sow(Integer(2))))

    def test_reap_with_tag_from_string(self):
        """Reap[Sow[1, myTag], myTag] parses to Reap(..., tag) with filter."""
        tag = Symbol('myTag')
        result = mathematica_to_sympy(
            "Reap[Sow[1, myTag], myTag]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Reap(Sow(Integer(1), tag), tag)

    # -- Head -----------------------------------------------------------------

    def test_head_integer_from_string(self):
        """Head[5] parses to Head(Integer(5))."""
        result = mathematica_to_sympy("Head[5]", custom_functions=_MATH_EXPR_FUNCS)
        assert result == Head(Integer(5))

    def test_head_list_from_string(self):
        """Head[List[1, 2]] parses to Head(List(Integer(1), Integer(2)))."""
        result = mathematica_to_sympy(
            "Head[List[1, 2]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Head(List(Integer(1), Integer(2)))

    def test_head_symbol_from_string(self):
        """Head[a] parses to Head(Symbol('a'))."""
        a = Symbol('a')
        result = mathematica_to_sympy(
            "Head[a]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Head(a)

    def test_head_add_from_string(self):
        """Head[Plus[a, b]] parses to Head(a + b)."""
        a, b = Symbol('a'), Symbol('b')
        result = mathematica_to_sympy(
            "Head[Plus[a, b]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        assert result == Head(a + b)

    # -- Nested constructs ----------------------------------------------------

    def test_nested_with_from_string(self):
        """With[List[Set[x, 2]], With[List[Set[y, 3]], Plus[x, y]]] parses correctly."""
        x, y = Symbol('x'), Symbol('y')
        result = mathematica_to_sympy(
            "With[List[Set[x, 2]], With[List[Set[y, 3]], Plus[x, y]]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        inner = With(List(Set(y, Integer(3))), x + y)
        outer = With(List(Set(x, Integer(2))), inner)
        assert_same_scope(result, outer)

    def test_if_in_with_from_string(self):
        """With[List[Set[x, 5]], If[Greater[x, 3], 100, 0]] parses correctly."""
        x = Symbol('x')
        result = mathematica_to_sympy(
            "With[List[Set[x, 5]], If[Greater[x, 3], 100, 0]]",
            custom_functions=_MATH_EXPR_FUNCS,
        )
        body = If(sympy.Gt(x, Integer(3)), Integer(100), Integer(0))
        assert_same_scope(result, With(List(Set(x, Integer(5))), body))


# ── _condition_holds lazy / short-circuit evaluation ─────────────────────────
# Regression guard: _condition_holds used to _eval the WHOLE test up-front, so a
# combined And could build/sort a structure embedding a non-Expr sentinel (a util
# returning the symbol False, e.g. DerivativeDivides) -> sympy sort crash
# ('bool' object has no attribute 'is_Float'). It now evaluates And/Or/Not
# lazily, operand-by-operand, matching Mathematica's short-circuiting.

def test_condition_holds_short_circuits_and_or():
    from sympy_wolfram.objects import _condition_holds, MathematicaExpr
    from sympy.logic.boolalg import And, Or, Boolean

    # Boolean base matters: real guard operands (MathematicaConstraint) inherit
    # Boolean via SymPyMatchingConstraint, and released sympy's And/Or reject
    # non-Boolean operands (as_Boolean) even with evaluate=False.
    class _Boom(MathematicaExpr, Boolean):
        def __new__(cls):
            return sympy.Expr.__new__(cls)
        def _evaluate(self, **kwargs):
            raise AssertionError("operand was evaluated despite short-circuit")

    # And(False, Boom): first operand False -> False, Boom must not be evaluated.
    assert _condition_holds(And(S.false, _Boom(), evaluate=False)) is False
    # Or(True, Boom): first operand True -> True, Boom must not be evaluated.
    assert _condition_holds(Or(S.true, _Boom(), evaluate=False)) is True


def test_doit_stays_unevaluated_when_evaluate_returns_none():
    from sympy_wolfram.objects import MathematicaExpr

    class _NoneNode(MathematicaExpr):
        def __new__(cls, arg):
            return sympy.Expr.__new__(cls, sympy.sympify(arg))
        def _evaluate(self, **kwargs):
            return None  # utility couldn't compute a value for this input

    node = _NoneNode(Symbol('x'))
    # doit must NOT return None (that would break an enclosing Add/Mul via
    # sympify(None)); it stays the unevaluated node instead.
    assert node.doit() is not None
    assert isinstance(node.doit(), _NoneNode)
    # and it must survive being embedded in arithmetic + doit'd
    expr = Symbol('y') * node
    assert expr.doit() is not None


def test_condition_holds_basic_connectives():
    from sympy_wolfram.objects import _condition_holds
    from sympy.logic.boolalg import And, Or, Not
    assert _condition_holds(And(S.true, S.true, evaluate=False)) is True
    assert _condition_holds(And(S.true, S.false, evaluate=False)) is False
    assert _condition_holds(Or(S.false, S.false, evaluate=False)) is False
    assert _condition_holds(Not(S.false)) is True
    assert _condition_holds(Not(S.true)) is False


# ── Condition (expr /; test) ─────────────────────────────────────────────────

def test_condition_holds_returns_body():
    from sympy_wolfram.objects import Condition
    assert Condition(Integer(5), S.true).doit() == 5


def test_condition_fails_raises_stopiteration():
    from sympy_wolfram.objects import Condition
    with pytest.raises(StopIteration):
        Condition(Integer(5), S.false).doit()


def test_condition_body_not_evaluated_when_test_fails():
    # The body must not be evaluated when the test fails (Mathematica semantics;
    # the default deep doit would have reduced the body first).
    from sympy_wolfram.objects import Condition, MathematicaExpr

    class _Boom(MathematicaExpr):
        def __new__(cls):
            return sympy.Expr.__new__(cls)
        def _evaluate(self, **kwargs):
            raise AssertionError("body evaluated despite failing test")

    with pytest.raises(StopIteration):
        Condition(_Boom(), S.false).doit()


def test_condition_set_in_test_binds_body():
    # Set[q, 7] inside the test binds q for the body (Mathematica side effect).
    from sympy_wolfram.objects import Condition, Set
    q = Symbol('q')
    cond = Condition(q + 1, Set(q, Integer(7)) > 0)
    assert cond.doit() == 8


# ── Scoping: With / Module / Block close over their own locals ────────────────
#
# Each construct alpha-renames its locals to Dummy symbols in __new__, so the
# scope is well defined the moment the node exists. Nothing outside has to know
# about it -- there is no "rename the locals first" pass to remember to call.

from sympy import Dummy

from sympy_wolfram.objects import Block, List, Module, Set, With

_SCOPING_CONSTRUCTS = [With, Module, Block]


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_a_local_is_bound_to_a_dummy(construct):
    a, b, x = Symbol('a'), Symbol('b'), Symbol('x')
    node = construct(List(Set(a, Integer(1))), a + b * x)
    local = node.args[0].args[0].args[0]
    assert isinstance(local, Dummy)
    assert local != a
    assert node.doit() == 1 + b * x


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_the_body_no_longer_mentions_the_original_symbol(construct):
    a, b = Symbol('a'), Symbol('b')
    node = construct(List(Set(a, Integer(1))), a + b)
    assert a not in node.args[1].free_symbols


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_an_outside_substitution_cannot_capture_a_local(construct):
    """The bug this design removes: substituting a value that happens to contain a
    symbol named like a local must not be captured by that local."""
    a, b, u = Symbol('a'), Symbol('b'), Symbol('u')
    template = construct(List(Set(a, u)), a * u)       # local a = u; body a*u
    substituted = template.subs(u, a + b)              # the value carries an 'a'
    assert substituted.doit() == (a + b)**2            # not (a+b)*(a+2*b)


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_a_binding_value_is_evaluated_in_the_enclosing_scope(construct):
    """``With[{a = f(a)}, ...]`` binds the local to the OUTER ``a``."""
    a, b = Symbol('a'), Symbol('b')
    node = construct(List(Set(a, a + b)), a)
    assert node.doit() == a + b                        # outer a survives in the value


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_rebuilding_the_node_does_not_rebind(construct):
    """SymPy rebuilds expressions constantly; re-binding each time would mint new
    dummies and detach the binder from its body."""
    a, b = Symbol('a'), Symbol('b')
    node = construct(List(Set(a, Integer(2))), a * b)
    rebuilt = node.func(*node.args)
    assert rebuilt == node
    assert rebuilt.doit() == node.doit() == 2 * b


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_two_nodes_with_the_same_local_name_are_independent(construct):
    a = Symbol('a')
    first = construct(List(Set(a, Integer(1))), a)
    second = construct(List(Set(a, Integer(2))), a)
    assert first.args[0] != second.args[0]
    assert first.doit() == 1 and second.doit() == 2


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_an_inner_scope_shadows_an_outer_one(construct):
    a, b = Symbol('a'), Symbol('b')
    inner = construct(List(Set(a, Integer(3))), a * b)
    outer = construct(List(Set(a, Integer(5))), a + inner)
    assert outer.doit() == 5 + 3 * b          # inner a=3 wins inside, outer a=5 outside


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_the_inner_local_is_a_distinct_dummy(construct):
    a, b = Symbol('a'), Symbol('b')
    inner = construct(List(Set(a, Integer(3))), a * b)
    outer = construct(List(Set(a, Integer(5))), a + inner)
    outer_local = outer.args[0].args[0].args[0]
    inner_local = inner.args[0].args[0].args[0]
    assert outer_local != inner_local


@pytest.mark.parametrize('construct', _SCOPING_CONSTRUCTS)
def test_several_locals_at_once(construct):
    a, b = Symbol('a'), Symbol('b')
    node = construct(List(Set(a, Integer(3)), Set(b, Integer(4))), a**2 + b**2)
    assert node.doit() == 25
    assert not ({a, b} & node.args[1].free_symbols)


def test_module_uninitialised_local_stays_a_dummy():
    a = Symbol('a')
    result = Module(List(a), a).doit()
    assert isinstance(result, Dummy)
    assert result != a


def test_module_uninitialised_locals_are_independent_across_nodes():
    a = Symbol('a')
    assert Module(List(a), a).doit() != Module(List(a), a).doit()


def test_expressions_without_a_scope_are_untouched():
    a, b, x = Symbol('a'), Symbol('b'), Symbol('x')
    expr = a * x + b
    assert expr.subs(a, Integer(2)) == 2 * x + b

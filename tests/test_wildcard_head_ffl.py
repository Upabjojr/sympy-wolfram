# -*- coding: utf-8 -*-
"""FFL -> SymPy conversion of WILDCARD FUNCTION HEADS.

A Rubi pattern ``F_[args]`` (a wildcard used as a function head) is rewritten by
the code generator into the FFL node ``WildHeadApp[F_, args...]``. This module
tests that the FFL converter turns that node into real ``WildHeadApp(...)`` code
and that the code evaluates back to the expected SymPy object.
"""
import pytest
import sympy

from sympy_matching.wild import WildHeadApp, WildHeadDeriv, WildSymbol
from sympy_wolfram.interpreter import FFLConverter
from sympy_wolfram.interpreter import ffl_to_sympy_short_code

# FFL shorthands
PAT = lambda n: ['Pattern', n, ['Blank']]          # noqa: E731  -- n_
OPT = lambda n: ['Optional', PAT(n)]               # noqa: E731  -- n_.


def convert(ffl, **kw):
    c = FFLConverter(**kw)
    c.reserved_symbols = {'x': 'x'}
    return c.convert(ffl, is_pattern=True)


class TestWildHeadAppCodeGeneration:

    def test_simple_single_argument(self):
        code = convert(['WildHeadApp', PAT('F'), PAT('v')])
        assert code == 'WildHeadApp(F_, v_)'

    def test_compound_plus_argument(self):
        code = convert(['WildHeadApp', PAT('F'),
                        ['Plus', OPT('a'), ['Times', OPT('b'), 'x']]])
        assert code.startswith('WildHeadApp(F_,')
        assert '_a_' in code and '_b_' in code and 'x' in code

    def test_compound_times_argument(self):
        code = convert(['WildHeadApp', PAT('F'), ['Times', OPT('d'), 'x']])
        assert code.startswith('WildHeadApp(F_,')
        assert '_d_' in code

    def test_multiple_arguments(self):
        code = convert(['WildHeadApp', PAT('F'), PAT('u'), PAT('v')])
        assert code == 'WildHeadApp(F_, u_, v_)'

    def test_head_wildcard_name_is_preserved(self):
        for name in ('F', 'G', 'trig'):
            code = convert(['WildHeadApp', PAT(name), PAT('v')])
            assert code == f'WildHeadApp({name}_, v_)'

    def test_nested_wild_head_apps(self):
        code = convert(['WildHeadApp', PAT('F'),
                        ['WildHeadApp', PAT('G'), PAT('v')]])
        assert code == 'WildHeadApp(F_, WildHeadApp(G_, v_))'


class TestWildHeadAppEvaluates:
    """The emitted code must evaluate to a genuine WildHeadApp."""

    def test_eval_namespace_exposes_wild_head_app(self):
        assert FFLConverter(reserved_symbols={'x': 'x'}).eval_ns['WildHeadApp'] is WildHeadApp

    def test_generated_code_evaluates_to_wild_head_app(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c.reserved_symbols = {'x': 'x'}
        code = c.convert(['WildHeadApp', PAT('F'), PAT('v')], is_pattern=True)
        ns = dict(c.eval_ns)
        ns.update({'F_': WildSymbol('F'), 'v_': WildSymbol('v')})
        obj = eval(code, ns)
        assert isinstance(obj, WildHeadApp)
        assert obj.head_wild.wildcard_name == 'F'
        assert obj.applied_args[0].wildcard_name == 'v'

    def test_short_code_api_roundtrip(self):
        """The high-level API used by the code generator emits the same node."""
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['WildHeadApp', PAT('F'), ['Plus', OPT('a'), ['Times', OPT('b'), 'x']]],
            reserved_symbols={'x': 'x'}, namespace=ns)
        assert code.startswith('WildHeadApp(F_,')
        obj = eval(code, {**ns})
        assert isinstance(obj, WildHeadApp)
        # the compound argument survived as a SymPy expression
        assert isinstance(obj.applied_args[0], sympy.Basic)


class TestOrdinaryConversionUnaffected:
    """A head that merely *looks* similar must not be treated specially."""

    def test_normal_function_head_still_converts_normally(self):
        assert convert(['Sin', 'x']) == 'sympy.sin(x)'

    def test_unknown_head_still_becomes_an_undefined_function(self):
        code = convert(['SomeUnknownHead', 'x'])
        assert "Function('SomeUnknownHead')" in code


class TestWildHeadDerivCodeGeneration:
    """``Derivative[n_][f_][x_]`` is rewritten by the code generator into the FFL
    node ``WildHeadDeriv[f_, x_, n_]``; the converter must emit that node."""

    def test_simple_wildcard_order(self):
        code = convert(['WildHeadDeriv', PAT('f'), 'x', PAT('n')])
        assert code.startswith('WildHeadDeriv(')
        assert 'f_' in code and 'n_' in code

    def test_evaluates_to_a_wild_head_deriv(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['WildHeadDeriv', PAT('f'), 'x', PAT('n')], reserved_symbols={'x': 'x'}, namespace=ns)
        obj = eval(code, {**ns})
        assert isinstance(obj, WildHeadDeriv)
        assert obj.head_wild.wildcard_name == 'f'
        assert obj.order.wildcard_name == 'n'
        assert obj.var == sympy.Symbol('x')

    def test_a_concrete_order_survives(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['WildHeadDeriv', PAT('f'), 'x', '2'], reserved_symbols={'x': 'x'}, namespace=ns)
        obj = eval(code, {**ns})
        assert isinstance(obj, WildHeadDeriv)
        assert obj.order == 2

    def test_nested_inside_a_product(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['Times', 'c', ['WildHeadDeriv', PAT('f'), 'x', PAT('n')]], reserved_symbols={'x': 'x'}, namespace=ns)
        obj = eval(code, {**ns})
        assert any(isinstance(a, WildHeadDeriv) for a in obj.args)

    def test_coexists_with_a_wild_head_app(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['Times', ['WildHeadApp', PAT('F'), 'x'],
             ['WildHeadDeriv', PAT('f'), 'x', PAT('n')]], reserved_symbols={'x': 'x'}, namespace=ns)
        obj = eval(code, {**ns})
        kinds = {type(a) for a in obj.args}
        assert WildHeadApp in kinds and WildHeadDeriv in kinds

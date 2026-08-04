# -*- coding: utf-8 -*-
"""Tests for sympy_wolfram.interpreter.FFLConverter."""
import pytest
from sympy import Symbol

from sympy_wolfram import mathematica_to_sympy_short_code
from sympy_wolfram.interpreter import FFLConverter


class TestAtoms:
    def test_integer(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('42') == 'Integer(42)'

    def test_negative_integer(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('-1') == 'Integer(-1)'

    def test_float_becomes_rational(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('0.5') == "Rational('0.5')"

    def test_constant_pi(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('Pi') == 'sympy.pi'

    def test_constant_e(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('E') == 'sympy.E'

    def test_constant_infinity(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('Infinity') == 'sympy.oo'

    def test_a_reserved_name_converts_to_its_identifier(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('x') == 'x'

    def test_plain_symbol(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('foo') == "Symbol('foo')"


class TestPatterns:
    def test_pattern_basic(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Pattern', 'a', ['Blank']], is_pattern=True)
        assert result == 'a_'
        assert 'a' in c.wildcards_non_optional

    def test_a_pattern_on_a_reserved_name_is_not_a_wildcard(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Pattern', 'x', ['Blank', 'Symbol']], is_pattern=True)
        assert result == 'x'

    def test_optional_pattern(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Optional', ['Pattern', 'b', ['Blank']]], is_pattern=True)
        assert result == '_b_'
        assert 'b' in c.wildcards_optional

    def test_wildcard_atom_reference(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_non_optional.add('m')
        assert c.convert('m') == 'm_'

    def test_optional_wildcard_atom_reference(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_optional.add('a')
        assert c.convert('a') == '_a_'


class TestArithmetic:
    def test_plus(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_non_optional = {'a', 'b'}
        result = c.convert(['Plus', 'a', 'b'])
        assert result == '(a_ + b_)'

    def test_times(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_non_optional = {'a'}
        result = c.convert(['Times', 'a', 'x'])
        assert result == '(a_ * x)'

    def test_times_strips_leading_one(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Times', '1', 'x'])
        assert result == 'x'

    def test_power(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Power', 'x', '2'])
        assert result == '(x)**(Integer(2))'

    def test_power_with_wildcard(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_optional = {'m'}
        result = c.convert(['Power', 'x', 'm'])
        assert result == '(x)**(_m_)'


class TestFunctions:
    def test_sin(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Sin', 'x'])
        assert result == 'sympy.sin(x)'

    def test_log(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Log', 'x'])
        assert result == 'sympy.log(x)'

    def test_arctan_one_arg(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['ArcTan', 'x'])
        assert result == 'sympy.atan(x)'

    def test_arctan_two_args(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['ArcTan', 'x', 'y'])
        # Mathematica ArcTan[x, y] = atan2(y, x)
        assert "sympy.atan2(Symbol('y'), x)" == result

    def test_hypergeometric2f1(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['Hypergeometric2F1', '1', '2', '3', 'x'])
        assert 'sympy.hyper' in result

    def test_unknown_head_fallback(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['SomeUnknown', 'x', '1'])
        assert result == "sympy.Function('SomeUnknown')(x, Integer(1))"


class TestList:
    def test_list_conversion(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['List', '1', '2', '3'])
        assert result == '[Integer(1), Integer(2), Integer(3)]'


class TestRemoveContent:
    def test_passthrough(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(['RemoveContent', 'x', 'x'])
        assert result == 'x'


class TestNonStringHead:
    def test_raises_value_error(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        with pytest.raises(ValueError, match="Non-string"):
            c.convert([['Pattern', 'F', ['Blank']], 'x'], is_pattern=True)


class TestFunctionAndSlot:
    """Tests for Mathematica Function[...] -> sympy.Lambda(...) conversion."""

    def test_function_single_slot(self):
        """Function[BinomialQ[#, x]] -> Lambda(Symbol('xi1'), BinomialQ(xi1, x))"""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["Function", ["BinomialQ", ["Slot", "1"], "x"]]
        result = c.convert(ffl)
        assert result == "Lambda(Symbol('xi1'), sympy.Function('BinomialQ')(Symbol('xi1'), x))"

    def test_function_in_everyq_context(self):
        """EveryQ[Function[BinomialQ[#, x]], P] — the actual Rubi pattern."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["EveryQ", ["Function", ["BinomialQ", ["Slot", "1"], "x"]], "P"]
        result = c.convert(ffl)
        assert "Lambda(Symbol('xi1')" in result
        assert "BinomialQ" in result
        assert "EveryQ" in result

    def test_function_multiple_slots(self):
        """Function[Plus[#1, #2]] -> Lambda with two params."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]]
        result = c.convert(ffl)
        assert "Lambda(" in result
        assert "Symbol('xi1')" in result
        assert "Symbol('xi2')" in result

    def test_function_named_param(self):
        """Function[p, body] -> Lambda(Symbol('p'), body)."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["Function", "p", ["BinomialQ", "p", "x"]]
        result = c.convert(ffl)
        assert result == "Lambda(Symbol('p'), sympy.Function('BinomialQ')(Symbol('p'), x))"

    def test_function_named_multi_params(self):
        """Function[{a, b}, Plus[a, b]] -> Lambda with tuple params."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["Function", ["List", "a", "b"], ["Plus", "a", "b"]]
        result = c.convert(ffl)
        assert "Lambda(" in result
        assert "Symbol('a')" in result
        assert "Symbol('b')" in result

    def test_slot_outside_function(self):
        """Slot[1] outside Function context still produces a symbol."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        result = c.convert(["Slot", "1"])
        assert result == "Symbol('xi1')"

    def test_slot_does_not_leak_between_functions(self):
        """Slot mapping is scoped to the enclosing Function."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        # Convert a Function — should set and restore _slot_vars
        ffl = ["Function", ["Slot", "1"]]
        c.convert(ffl)
        # After conversion, _slot_vars should be empty again
        assert c._slot_vars == {}

    def test_function_no_slot_defaults_to_xi1(self):
        """Function[x] with no Slot still produces Lambda with xi1 param."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["Function", "x"]
        result = c.convert(ffl)
        assert "Lambda(Symbol('xi1'), x)" == result



class TestBooleanAtoms:
    """True/False Mathematica atoms must map to sympy.true/sympy.false."""

    def test_true_atom(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('True') == 'sympy.true'

    def test_false_atom(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert('False') == 'sympy.false'

    def test_true_as_function_arg(self):
        """True passed as an argument to a function call."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["SomeFunc", "x", "True"]
        result = c.convert(ffl)
        assert 'sympy.true' in result
        assert "Symbol('True')" not in result

    def test_false_as_function_arg(self):
        """False passed as an argument to a function call."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        ffl = ["SomeFunc", "x", "False"]
        result = c.convert(ffl)
        assert 'sympy.false' in result
        assert "Symbol('False')" not in result

    def test_true_not_treated_as_wildcard(self):
        """True must not become a WildSymbol even if in wildcards set."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c._wildcards_non_optional.add('True')
        # Despite being in wildcards, CONSTANT_MAP should take priority
        result = c.convert('True')
        assert result == 'sympy.true'


class TestOptionalOnTheFixedVariable:
    """``x_.`` where ``x`` is ALSO the fixed variable (bound by ``x_Symbol``).

    Both bind the same name, so the "absent" branch would have to give ``x`` the
    Times identity 1, which then fails ``x_Symbol``. The optional branch is thus
    unreachable and the factor is really mandatory -- it must convert to the plain
    fixed variable, exactly as a non-optional ``x_`` does.

    Verified in Mathematica: ``g[x_.*h[x_], x_Symbol]`` matches ``g[z h[z], z]``
    but NOT ``g[h[z], z]``; with a differently-named ``u_.`` the absent branch does
    match, binding ``u -> 1``.
    """

    def test_optional_on_a_reserved_name_becomes_the_plain_identifier(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Optional', ['Pattern', 'x', ['Blank']]],
                         is_pattern=True) == 'x'

    def test_it_does_not_declare_an_optional_wildcard(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        c.convert(['Optional', ['Pattern', 'x', ['Blank']]], is_pattern=True)
        assert 'x' not in c._wildcards_optional
        assert not any('_x_' in d for d in c._wild_defs)

    def test_it_agrees_with_the_non_optional_form(self):
        """Pattern[x, Blank] already collapses to the fixed var; Optional must match."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        plain = c.convert(['Pattern', 'x', ['Blank']], is_pattern=True)
        opt = c.convert(['Optional', ['Pattern', 'x', ['Blank']]], is_pattern=True)
        assert plain == opt == 'x'

    def test_a_differently_named_optional_is_still_optional(self):
        """The control case: only the fixed variable's own name is affected."""
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Optional', ['Pattern', 'u', ['Blank']]],
                         is_pattern=True) == '_u_'
        assert 'u' in c._wildcards_optional

    def test_inside_a_product(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        code = c.convert(['Times', ['Optional', ['Pattern', 'x', ['Blank']]],
                          ['Power', ['Pattern', 'c', ['Blank']], '-1']],
                         is_pattern=True)
        assert '_x_' not in code
        assert 'x' in code

    def test_a_different_reserved_name_is_honoured(self):
        c = FFLConverter(reserved_symbols={'t': 'x'})
        assert c.convert(['Optional', ['Pattern', 't', ['Blank']]],
                         is_pattern=True) == 'x'
        assert 't' not in c._wildcards_optional


# =============================================================================
# reserved_symbols / namespace / str_printer
#
# These three replace the old `fixed_var`, the returned namespace, and the
# hard-wired printer. sympy_wolfram has no notion of an "integration variable" --
# the CALLER declares which names are externally bound.
# =============================================================================

from sympy.printing.str import StrPrinter

import sympy

from sympy_wolfram.interpreter import ffl_to_sympy_code, ffl_to_sympy_short_code


class TestReservedSymbols:

    def test_a_reserved_name_is_not_a_wildcard(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Pattern', 'x', ['Blank']], is_pattern=True) == 'x'
        assert 'x' not in c._wildcards_non_optional

    def test_an_unreserved_name_still_becomes_a_wildcard(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Pattern', 'm', ['Blank']], is_pattern=True) == 'm_'

    def test_with_no_reserved_symbols_every_name_is_free(self):
        """The default is empty: nothing is special, not even 'x'."""
        c = FFLConverter()
        assert c.convert('x') == "Symbol('x')"
        assert c.convert(['Pattern', 'x', ['Blank']], is_pattern=True) == 'x_'

    def test_the_reserved_name_maps_to_its_own_identifier(self):
        """The Wolfram name and the emitted identifier need not match."""
        c = FFLConverter(reserved_symbols={'t': 'x'})
        assert c.convert('t') == 'x'
        assert c.convert(['Pattern', 't', ['Blank']], is_pattern=True) == 'x'

    def test_several_reserved_names_at_once(self):
        c = FFLConverter(reserved_symbols={'u': 'u', 'v': 'v'})
        assert c.convert('u') == 'u'
        assert c.convert('v') == 'v'
        assert c.convert('w') == "Symbol('w')"

    def test_a_reserved_identifier_evaluates_to_its_wolfram_symbol(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_code(['Power', 't', '2'], {'t': 'x'}, ns)
        assert code == '(x)**(Integer(2))'
        assert eval(code, ns) == Symbol('t')**2


class TestNamespaceIsFilledInPlace:

    def test_the_passed_dict_can_eval_the_returned_code(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_code(['Sin', 'x'], {'x': 'x'}, ns)
        assert eval(code, ns) == sympy.sin(Symbol('x'))

    def test_discovered_wildcards_land_in_the_namespace(self):
        ns = {}
        code, _defs, _syms = ffl_to_sympy_short_code(
            ['Power', 'x', ['Pattern', 'm', ['Blank']]], {'x': 'x'}, ns)
        assert 'm_' in ns
        assert eval(code, ns) is not None

    def test_caller_entries_take_precedence_over_the_base_names(self):
        marker = Symbol('my_own_log')
        ns = {'log': marker}
        ffl_to_sympy_code(['Sin', 'x'], {'x': 'x'}, ns)
        assert ns['log'] is marker

    def test_a_fresh_dict_per_call_keeps_calls_independent(self):
        ns1, ns2 = {}, {}
        ffl_to_sympy_code(['Pattern', 'm', ['Blank']], {'x': 'x'}, ns1)
        ffl_to_sympy_code(['Pattern', 'q', ['Blank']], {'x': 'x'}, ns2)
        assert 'm_' in ns1 and 'm_' not in ns2
        assert 'q_' in ns2 and 'q_' not in ns1

    def test_omitting_the_namespace_still_works(self):
        code, _defs, _syms = ffl_to_sympy_code(['Sin', 'x'], {'x': 'x'})
        assert code == 'sympy.sin(x)'


class TestStrPrinter:

    def test_default_printer_shortens_the_code(self):
        ns = {}
        short, _defs, _syms = ffl_to_sympy_short_code(
            ['Power', 'x', '2'], {'x': 'x'}, ns)
        assert short == 'x**2'          # not '(x)**(Integer(2))'

    def test_default_printer_uses_the_wildcard_convention(self):
        ns = {}
        short, _defs, _syms = ffl_to_sympy_short_code(
            ['Pattern', 'm', ['Blank']], {'x': 'x'}, ns)
        assert short == 'm_'

    def test_a_custom_printer_is_used(self):
        class _Shouty(StrPrinter):
            def _print_Symbol(self, expr):
                return expr.name.upper()

        ns = {'X': Symbol('x')}
        short, _defs, _syms = ffl_to_sympy_short_code(
            ['Power', 'x', '2'], {'x': 'x'}, ns, str_printer=_Shouty())
        assert short == 'X**2'
        assert eval(short, ns) == Symbol('x')**2

    def test_a_printer_whose_output_does_not_round_trip_is_rejected(self):
        """The shortening must never change meaning: if the printed form does not
        evaluate back to the same object, the original code is kept."""
        class _Liar(StrPrinter):
            def _print_Symbol(self, expr):
                return 'not_the_same_symbol'

        ns = {'not_the_same_symbol': Symbol('something_else')}
        short, _defs, _syms = ffl_to_sympy_short_code(
            ['Power', 'x', '2'], {'x': 'x'}, ns, str_printer=_Liar())
        assert short == '(x)**(Integer(2))'   # unchanged

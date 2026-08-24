# -*- coding: utf-8 -*-
"""Wolfram heads must translate to a name that EVALUATES in generated code.

Two translation modes are tested here:

* the DEFAULT mode, which is FROZEN: it is what rubi_integrate generates its rule
  set with, and any change to it changes those rules (in a rule the arguments are
  wildcards that may match any expression, so an eager SymPy function might evaluate
  them as if they were constants);
* ``standard_sympy=True``, which translates every head with a standard SymPy
  equivalent -- meant for plain expressions such as the Rubi test-suite answers.
"""
import pytest
import sympy
from sympy import Symbol, symbols

from sympy_wolfram import ffl_to_sympy_short_code
from sympy_wolfram.interpreter import FFLConverter, ffl_to_sympy_code
from sympy_wolfram.objects import rewrite_as_standard_sympy

x = Symbol('x')
a, b, c, m, n, s = symbols('a b c m n s')


def _eval(ffl, **kw):
    ns = {}
    code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'}, ns, standard_sympy=True, **kw)
    assert 'sympy.Function(' not in code, code
    return eval(code, ns)


# Heads found in the Rubi test-suite answers, with the SymPy object each must
# evaluate to. They were emitted as ``sympy.Function('Head')`` placeholders by an
# older generator, leaving 16% of the suite's expected answers unusable.
RUBI_SUITE_HEADS = [
    (['PolyLog', '2', 'x'], sympy.polylog(2, x)),
    (['Erf', 'x'], sympy.erf(x)),
    (['Erfi', 'x'], sympy.erfi(x)),
    (['Erfc', 'x'], sympy.erfc(x)),
    (['FresnelS', 'x'], sympy.fresnels(x)),
    (['FresnelC', 'x'], sympy.fresnelc(x)),
    (['SinIntegral', 'x'], sympy.Si(x)),
    (['CosIntegral', 'x'], sympy.Ci(x)),
    (['SinhIntegral', 'x'], sympy.Shi(x)),
    (['CoshIntegral', 'x'], sympy.Chi(x)),
    (['LogGamma', 'x'], sympy.loggamma(x)),
    (['EllipticE', 'm'], sympy.elliptic_e(m)),
    (['EllipticE', 'x', 'm'], sympy.elliptic_e(x, m)),
    (['EllipticF', 'x', 'm'], sympy.elliptic_f(x, m)),
    (['EllipticK', 'm'], sympy.elliptic_k(m)),
    (['EllipticPi', 'n', 'm'], sympy.elliptic_pi(n, m)),
    (['EllipticPi', 'n', 'x', 'm'], sympy.elliptic_pi(n, x, m)),
    (['Hypergeometric2F1', 'a', 'b', 'c', 'x'], sympy.hyper([a, b], [c], x)),
    (['HypergeometricPFQ', ['List', 'a', 'b'], ['List', 'c'], 'x'],
     sympy.hyper([a, b], [c], x)),
    (['HypergeometricPFQ', ['List', 'a', 'b', 'c'], ['List', 'm', 'n'], 'x'],
     sympy.hyper([a, b, c], [m, n], x)),
    (['AppellF1', 'a', 'b', 'c', 'm', 'x', 'n'], sympy.appellf1(a, b, c, m, x, n)),
    (['Expand', ['Times', 'x', ['Plus', 'x', '1']]], x**2 + x),
]

# Heads this package implements as its own node: the node is what gets emitted,
# and ``rewrite_as_standard_sympy`` (applied automatically in standard mode)
# gives the SymPy equivalent.
RUBI_SUITE_NODE_HEADS = [
    (['Gamma', 'x'], sympy.gamma(x)),
    (['Gamma', 'a', 'x'], sympy.uppergamma(a, x)),
    (['ProductLog', 'x'], sympy.LambertW(x)),
    (['ProductLog', '-1', 'x'], sympy.LambertW(x, -1)),
    (['ExpIntegralEi', 'x'], sympy.Ei(x)),
    (['ExpIntegralE', 'n', 'x'], sympy.expint(n, x)),
    (['LogIntegral', 'x'], sympy.li(x)),
    (['Zeta', 's'], sympy.zeta(s)),
    (['Zeta', 's', 'a'], sympy.zeta(s, a)),
    (['PolyGamma', 'n', 'x'], sympy.polygamma(n, x)),
    (['BesselJ', 'n', 'x'], sympy.besselj(n, x)),
    (['Factorial', 'n'], sympy.factorial(n)),
]


class TestRubiSuiteHeadsTranslate:
    def test_sympy_heads_evaluate_to_the_sympy_function(self):
        for ffl, expected in RUBI_SUITE_HEADS:
            assert _eval(ffl) == expected, ffl

    def test_node_heads_are_rewritten_to_standard_sympy(self):
        for ffl, expected in RUBI_SUITE_NODE_HEADS:
            assert _eval(ffl) == expected, ffl

    def test_node_heads_stay_nodes_when_rewrite_is_identity(self):
        from sympy_wolfram.objects import MathematicaExpr
        for ffl, _ in RUBI_SUITE_NODE_HEADS:
            got = _eval(ffl, rewrite=lambda e: e)
            assert isinstance(got, MathematicaExpr) and type(got).__name__ == ffl[0], ffl

    def test_rewritten_code_is_shortened_not_verbose(self):
        """Every SymPy name a rewrite can produce must be in the shortening namespace,
        or the round-trip fails and the verbose node code is emitted instead."""
        for ffl, _ in RUBI_SUITE_NODE_HEADS + RUBI_SUITE_HEADS:
            code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'}, standard_sympy=True)
            assert "Symbol('" not in code, (ffl, code)


class TestDefaultModeIsFrozen:
    """The default translation is what rubi_integrate's rule set is generated with.
    Everything standard-SymPy mode adds must be invisible here."""

    # FFLConverter.generated_code_sympy_names() on the master branch before the
    # standard-SymPy mode existed. rubi_integrate builds the ``from sympy import``
    # header of every rule module from this set, so adding a name re-wraps the
    # header of most of the 245 generated files.
    MASTER_GENERATED_NAMES = [
        'Abs', 'Chi', 'Ci', 'Ei', 'Eq', 'Ge', 'Gt', 'I', 'LambertW', 'Le', 'Lt', 'Ne',
        'Shi', 'Si', 'acos', 'acosh', 'acot', 'acoth', 'acsc', 'acsch', 'appellf1',
        'asec', 'asech', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'besselj', 'cos',
        'cosh', 'cot', 'coth', 'csc', 'csch', 'denom', 'diff', 'elliptic_e',
        'elliptic_f', 'erf', 'erfc', 'erfi', 'exp', 'expint', 'factorial', 'floor',
        'frac', 'fresnelc', 'fresnels', 'gamma', 'hyper', 'li', 'log', 'loggamma',
        'oo', 'pi', 'polygamma', 'polylog', 'root', 'sec', 'sech', 'simplify', 'sin',
        'sinh', 'sqrt', 'tan', 'tanh', 'uppergamma', 'zeta',
    ]

    def test_generated_name_set_is_unchanged(self):
        assert sorted(FFLConverter.generated_code_sympy_names()) == self.MASTER_GENERATED_NAMES

    def test_standard_mode_only_adds_names(self):
        extra = set(FFLConverter.generated_code_sympy_names(standard_sympy=True)) - set(self.MASTER_GENERATED_NAMES)
        assert extra == {'elliptic_k', 'elliptic_pi', 'expand', 'Derivative', 'Subs', 'EulerGamma'}

    def test_elliptic_pi_stays_a_node_that_does_not_self_translate(self):
        from sympy_wolfram.mathematica_functions import EllipticPi
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['EllipticPi', 'n', 'x', 'm']) == "EllipticPi(Symbol('n'), x, Symbol('m'))"
        # rubi_integrate keeps the node as a deferred placeholder BECAUSE of this.
        assert 'rewrite_as_standard_sympy' not in EllipticPi.__dict__
        assert rewrite_as_standard_sympy(EllipticPi(n, m)) == EllipticPi(n, m)

    def test_two_argument_log_keeps_the_historical_order(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Log', '2', 'x']) == 'sympy.log(Integer(2), x)'

    def test_elliptic_k_and_derivative_heads_are_untranslated(self):
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['EllipticK', 'm']) == "sympy.Function('EllipticK')(Symbol('m'))"
        with pytest.raises(ValueError, match='Non-string function head'):
            c.convert([[['Derivative', '1'], 'f'], 'x'])

    def test_shortening_printer_is_unchanged(self):
        # Unknown heads, Euler's number and shadowed symbols print bare, so the
        # round-trip fails and the verbose code is kept -- as before.
        ffl = ['Plus', ['Unintegrable', 'x', 'x'], ['Times', '2', 'E', 'D']]
        code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'})
        assert code == "(sympy.Function('Unintegrable')(x, x) + (Integer(2) * sympy.E * Symbol('D')))"

    def test_expand_is_a_deferred_node(self):
        from sympy_wolfram.mathematica_functions import Expand
        c = FFLConverter(reserved_symbols={'x': 'x'})
        assert c.convert(['Expand', 'x', 'x']) == 'Expand(x, x)'
        e = Expand(x * (x + 1), x)
        assert e.args == (x * (x + 1), x) and e.doit() == x**2 + x


class TestHypergeometricPFQ:
    def test_list_arguments_become_hyper_parameter_lists(self):
        # In BOTH modes: the placeholder crashed on the Python-list arguments (every
        # generated test-suite module containing a pFq failed to import), and
        # ``hyper`` was already a generated name, so nothing else changes.
        for flag in (False, True):
            c = FFLConverter(reserved_symbols={'x': 'x'}, standard_sympy=flag)
            code = c.convert(['HypergeometricPFQ', ['List', 'a', 'b'], ['List', 'c'], 'x'])
            assert code == "sympy.hyper([Symbol('a'), Symbol('b')], [Symbol('c')], x)"

    def test_the_old_placeholder_could_not_even_be_evaluated(self):
        with pytest.raises(AttributeError):
            sympy.Function('HypergeometricPFQ')([a, b], [c], x)


class TestLogArgumentOrder:
    def test_one_argument_log(self):
        assert _eval(['Log', 'x']) == sympy.log(x)

    def test_two_argument_log_is_base_first_in_mathematica(self):
        # Log[b, z] == log_b(z); sympy.log(z, b) takes the arguments the other way.
        assert _eval(['Log', '2', 'x']) == sympy.log(x, 2)
        assert _eval(['Log', '2', 'x']) == sympy.log(x) / sympy.log(2)

    def test_numeric_two_argument_log(self):
        code, _, _ = ffl_to_sympy_code(['Log', '2', '8'], standard_sympy=True)
        assert eval(code, {'sympy': sympy, 'Integer': sympy.Integer}) == 3

    def test_custom_function_override_still_wins(self):
        c = FFLConverter(reserved_symbols={'x': 'x'}, standard_sympy=True,
                         custom_functions={'Log': ('mylog', lambda *a: a)})
        assert c.convert(['Log', '2', 'x']) == "mylog(Integer(2), x)"


class TestExpandInStandardMode:
    def test_one_and_two_argument_forms(self):
        # Rubi: Int[Expand[Sin[e + f*x]^m*(a + b*Tan[e + f*x])^n, x], x]
        assert _eval(['Expand', ['Times', 'a', ['Plus', 'x', '1']]]) == a * x + a
        assert _eval(['Expand', ['Times', 'a', ['Plus', 'x', '1']], 'x']) == a * x + a


class TestWildcardArgumentsAreNotEvaluatedAway:
    """Even in standard mode the emitted SymPy function must not treat wildcard
    arguments as constants and evaluate them."""

    @staticmethod
    def _pattern(ffl):
        ns = {}
        code, defs, _ = ffl_to_sympy_code(ffl, {'x': 'x'}, ns, standard_sympy=True)
        for d in defs:
            exec(d, ns)
        return eval(code, ns), ns

    P = staticmethod(lambda name: ['Pattern', name, ['Blank']])

    def test_elliptic_k_of_a_wildcard_stays_a_call(self):
        got, ns = self._pattern(['EllipticK', self.P('m')])
        assert isinstance(got, sympy.elliptic_k)
        assert got.args == (ns['m_'],)

    def test_elliptic_pi_of_wildcards_stays_a_call(self):
        got, ns = self._pattern(['EllipticPi', self.P('n'), self.P('m')])
        assert isinstance(got, sympy.elliptic_pi)
        assert got.args == (ns['n_'], ns['m_'])

    def test_hyper_of_wildcards_keeps_its_parameter_lists(self):
        got, ns = self._pattern(['HypergeometricPFQ', ['List', self.P('a'), self.P('b')],
                                 ['List', self.P('c')], self.P('z')])
        assert isinstance(got, sympy.hyper)
        assert got.ap == (ns['a_'], ns['b_'])
        assert got.bq == (ns['c_'],)
        assert got.argument == ns['z_']

    def test_two_argument_log_matches_mathematicas_own_evaluation(self):
        # Mathematica itself evaluates Log[b, z] to Log[z]/Log[b], so the quotient
        # is the faithful structure -- with the wildcards intact inside it.
        got, ns = self._pattern(['Log', self.P('b'), self.P('z')])
        assert got == sympy.log(ns['z_']) / sympy.log(ns['b_'])


class TestShorteningSurvivesUnknownHeads:
    """A Rubi marker (Unintegrable[...]) or an arbitrary F[x] in an answer used to make
    the shortening round-trip raise NameError, so the whole answer stayed verbose --
    and the rewrite to standard SymPy was silently lost with it."""

    def test_marker_head_is_qualified_and_the_rest_is_rewritten(self):
        ffl = ['Plus', ['Unintegrable', ['Power', 'x', 'x'], 'x'], ['Gamma', 'a', 'x']]
        ns = {}
        code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'}, ns, standard_sympy=True)
        assert code == "sympy.Function('Unintegrable')(x**x, x) + uppergamma(a, x)"
        assert eval(code, ns) == sympy.uppergamma(a, x) + sympy.Function('Unintegrable')(x**x, x)

    def test_registered_placeholders_keep_their_bare_call_form(self):
        # Rubi utilities / constraint predicates are registered by the caller as
        # unevaluated Function placeholders and must still print as FreeQ(a, x).
        ns = {'FreeQ': sympy.Function('FreeQ')}
        code, _, _ = ffl_to_sympy_short_code(['FreeQ', 'a', 'x'], {'x': 'x'}, ns, standard_sympy=True,
                                             custom_functions={'FreeQ': ('FreeQ', ns['FreeQ'])})
        assert code == 'FreeQ(a, x)'

    def test_euler_gamma_is_in_the_shortening_namespace(self):
        ffl = ['Plus', ['Times', 'EulerGamma', ['Log', 'x']], ['ExpIntegralE', '2', 'x']]
        ns = {}
        code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'}, ns, standard_sympy=True)
        assert code == 'EulerGamma*log(x) + expint(2, x)'

    def test_eulers_number_prints_qualified(self):
        ns = {}
        code, _, _ = ffl_to_sympy_short_code(['Times', '2', 'E', ['Power', 'r', '2']], {}, ns,
                                             standard_sympy=True)
        assert code == '2*sympy.E*r**2'
        assert eval(code, ns) == 2 * sympy.E * Symbol('r')**2

    def test_coefficient_named_like_a_wolfram_node_prints_as_symbol(self):
        # Rubi's (A + B x + C x^2 + D x^3): ``D`` is also the Wolfram D node, which
        # is what the bare name resolves to in the shortening namespace.
        ns = {}
        code, _, _ = ffl_to_sympy_short_code(['Plus', 'A', ['Times', 'D', 'x']], {'x': 'x'}, ns,
                                             standard_sympy=True)
        assert code == "A + Symbol('D')*x"
        assert eval(code, ns) == Symbol('A') + Symbol('D') * x


class TestDerivativeHead:
    @staticmethod
    def _eval_with_f(ffl):
        # The arbitrary f is an undefined function, so (unlike the other heads
        # here) sympy.Function('f') IS the correct emission.
        ns = {}
        code, _, _ = ffl_to_sympy_short_code(ffl, {'x': 'x'}, ns, standard_sympy=True)
        return eval(code, ns)

    def test_first_derivative_of_an_arbitrary_function(self):
        # f'[x] parses to [[['Derivative', '1'], 'f'], 'x']
        f = sympy.Function('f')
        assert self._eval_with_f([[['Derivative', '1'], 'f'], 'x']) == sympy.Derivative(f(x), x)

    def test_higher_order_is_shortened_and_round_trips(self):
        ns = {}
        code, _, _ = ffl_to_sympy_short_code([[['Derivative', '2'], 'f'], 'x'], {'x': 'x'}, ns,
                                             standard_sympy=True)
        assert code == "Derivative(sympy.Function('f')(x), (x, 2))"
        assert eval(code, ns) == sympy.Derivative(sympy.Function('f')(x), (x, 2))

    def test_order_zero_is_the_function_itself(self):
        assert self._eval_with_f([[['Derivative', '0'], 'f'], 'x']) == sympy.Function('f')(x)

    def test_derivative_at_a_non_symbol_argument_is_a_subs(self):
        # 8.10 Formal derivatives: f'[Sin[x]]. SymPy cannot differentiate with respect
        # to sin(x); the faithful form is the one diff() itself produces.
        f = sympy.Function('f')
        got = self._eval_with_f([[['Derivative', '1'], 'f'], ['Sin', 'x']])
        assert isinstance(got, sympy.Subs)
        assert got == sympy.diff(f(sympy.sin(x)), x) / sympy.cos(x)

    def test_subs_form_is_shortened_and_declares_its_dummy(self):
        ns = {}
        code, _, symbols_ = ffl_to_sympy_short_code(
            [[['Derivative', '2'], 'f'], ['Power', 'x', '2']], {'x': 'x'}, ns, standard_sympy=True)
        assert code == "Subs(Derivative(sympy.Function('f')(_xi_1), (_xi_1, 2)), _xi_1, x**2)"
        assert '_xi_1' in symbols_ and eval(code, ns).doit() == eval(code, ns)

    def test_symbolic_and_negative_orders_are_still_rejected(self):
        c = FFLConverter(reserved_symbols={'x': 'x'}, standard_sympy=True)
        for order in ('m', '-1'):
            with pytest.raises(ValueError, match='Non-string function head'):
                c.convert([[['Derivative', order], 'f'], 'x'])

# -*- coding: utf-8 -*-
"""Standard Wolfram-language function nodes (deferred, SymPy-backed).

These model Mathematica *standard-library* functions (``GCD``, ``Sign``, ``Floor``,
``Together``, ``ProductLog``, ``LeafCount``, …) as :class:`~sympy_wolfram.objects.MathematicaExpr`
subclasses.  They used to live in ``rubi_integrate.utils.rubi_utils``, but they are not
Rubi-specific — they belong to the generic Wolfram layer, so they live here and are
re-exported from ``rubi_utils`` for backward compatibility.

Only functions whose evaluation depends *solely* on SymPy (directly, or via the
self-contained eager helpers in :mod:`sympy_wolfram.functions_eager`) live here.
That includes ``First``/``Rest``/``Exponent``/``Apart``/``Part``: their seemingly
Rubi-coupled bodies (``SumQ``/``Sort``/``PolynomialQ``/``RationalFunctionQ``/
``Util_Part``) turned out to be generic SymPy operations, so those deferred classes
live HERE too, not in ``rubi_integrate``.

Note: the deferred class keeps the bare Mathematica name while its eager counterpart
is prefixed with ``eager_`` and lives in a different module — the node class
``LeafCount`` is here, the eager function ``eager_LeafCount`` is in
:mod:`sympy_wolfram.functions_eager`.
"""
import sympy
from sympy import Expr, Integer, S

from sympy_wolfram.objects import List, MathematicaExpr
from sympy_wolfram import functions_eager as _eager

# Above this many terms, expanding a Sum term-by-term costs more than it buys;
# hand it back to sympy.Sum, which can still find a closed form.
_SUM_EXPANSION_LIMIT = 200


# ---------------------------------------------------------------------------
# Bucket A — evaluation depends solely on SymPy.
# ---------------------------------------------------------------------------

class Coefficient(MathematicaExpr):
    """Mathematica Coefficient[expr, x, n] -> coefficient of x^n in expr."""

    def __new__(cls, expr, x, n=S.One):
        n = sympy.sympify(n)
        return Expr.__new__(cls, expr, x, n)

    def _evaluate(self, **kwargs):
        expr, x, n = self.args
        if n == S.Zero:
            return expr.coeff(x, 0)
        try:
            return sympy.Poly(expr, x).nth(int(n))
        except Exception:
            return expr.coeff(x, int(n))


class PolynomialQuotient(MathematicaExpr):
    """Mathematica PolynomialQuotient[p, q, x] -> quotient of p/q in x."""

    def __new__(cls, p, q, x):
        return Expr.__new__(cls, p, q, x)

    def _evaluate(self, **kwargs):
        # Delegate to the eager helper (see functions_eager.PolynomialQuotient), which
        # handles the RATIONAL-p case Rubi relies on (Pq*(c x)^m with m<0). The old inline
        # sympy.quo(...) here returned 0 on such inputs -- e.g.
        # PolynomialQuotient[(A+Bx)/x^2, a+b x^2] -> 0 -- silently zeroing whole integrals.
        from sympy_wolfram.functions_eager import eager_PolynomialQuotient
        return eager_PolynomialQuotient(*self.args)


class PolynomialRemainder(MathematicaExpr):
    """Mathematica PolynomialRemainder[p, q, x] -> remainder of p/q in x."""

    def __new__(cls, p, q, x):
        return Expr.__new__(cls, p, q, x)

    def _evaluate(self, **kwargs):
        # Delegate to the eager helper (see functions_eager.PolynomialRemainder), which
        # reduces a RATIONAL p modulo q; the old inline sympy.rem(...) returned the whole
        # input p on such Rubi inputs, breaking the rules that use the remainder's coeffs.
        from sympy_wolfram.functions_eager import eager_PolynomialRemainder
        return eager_PolynomialRemainder(*self.args)


class Rule(MathematicaExpr):
    """Mathematica Rule[lhs, rhs] — a (lhs -> rhs) substitution descriptor.

    Used as argument to ReplaceAll. _evaluate returns self because Rule
    is structural rather than a reducible expression.
    """

    def __new__(cls, lhs, rhs):
        return Expr.__new__(cls, lhs, rhs)

    def _evaluate(self, **kwargs):
        return self


class ReplaceAll(MathematicaExpr):
    """Mathematica ReplaceAll[expr, Rule[lhs, rhs]] — substitute lhs -> rhs."""

    def __new__(cls, expr, rule):
        return Expr.__new__(cls, expr, rule)

    def _evaluate(self, **kwargs):
        expr, rule = self.args
        if isinstance(rule, Rule):
            lhs, rhs = rule.args
            return expr.subs(lhs, rhs)
        # Mathematica also accepts a LIST of rules ({aa->a, bb->b, ...}), applied
        # simultaneously in one pass. This branch used to fall through and return
        # expr UNCHANGED, so rules built on the Module[{aa,bb,cc}, ...
        # ReplaceAll[..., {aa->a, bb->b, cc->c}]] idiom (e.g. 1.2.2.3 #86) leaked
        # their scoped dummies straight into the antiderivative.
        if isinstance(rule, (List, list, tuple)):
            items = rule.args if isinstance(rule, List) else rule
            pairs = [tuple(r.args) for r in items if isinstance(r, Rule)]
            if pairs and len(pairs) == len(list(items)):
                return expr.subs(pairs, simultaneous=True)
        return expr


def _expand_sum(expr, limits):
    """Mathematica ``Sum`` over CONCRETE bounds, or None if they are not concrete.

    Mathematica steps the iterator by 1 starting EXACTLY at ``imin`` and stops at
    the last value <= ``imax``; the lower bound is NOT rounded. Verified on
    Mathematica 12.2: ``Sum[k,{k,1/2,7/2}]`` = 8 (k = 1/2,3/2,5/2,7/2) -- rounding
    imin up would give 6 -- and ``Sum[x^(3k),{k,0,11/3}]`` = 1+x^3+x^6+x^9.
    sympy.Sum leaves any fractional-bound sum unevaluated, so expand explicitly.
    """
    if not (isinstance(limits, List) and len(limits.args) == 3):
        return None
    i, imin, imax = limits.args
    if not (getattr(imin, "is_number", False) and getattr(imax, "is_number", False)):
        return None
    try:
        steps = sympy.floor(imax - imin)
        if steps.is_negative:
            return S.Zero
        count = int(steps)
    except (TypeError, ValueError):
        return None
    if count > _SUM_EXPANSION_LIMIT:
        return sympy.Sum(expr, (i, imin, imin + count)).doit()
    return sympy.Add(*[expr.subs(i, imin + t) for t in range(count + 1)])


class SumWolfram(MathematicaExpr):
    """Mathematica Sum[expr, {i, imin, imax}] — symbolic summation.

    Delegates to sympy.Sum when limits is a List of three elements.
    """

    def __new__(cls, expr, limits):
        # The generated Rubi rules spell the iterator spec as a PYTHON list --
        # Sum(..., [ii, 0, n/2 - 1]). Storing that raw list in .args breaks SymPy's
        # invariant that every arg is a Basic, so ANY generic traversal of the node
        # (free_symbols, xreplace, subs) raises AttributeError deep inside SymPy.
        # That stayed hidden while the two constraints using this node were being
        # dropped by the matcher; enforcing them surfaced it. Coerce to the Wolfram
        # List node -- which is also the form _evaluate below expects, so the
        # documented floor/ceil handling now actually runs for these rules.
        if isinstance(limits, (list, tuple)):
            limits = List(*limits)
        # NOTE: do NOT expand here. Rubi's `Module[{k,u}, u = f[k]; Sum[u,{k,1,n}]]`
        # idiom builds the Sum while its summand is still the bare local `u`; the
        # binding arrives afterwards from the enclosing Set. Expanding at
        # construction would sum `u` with itself n times and lose the iterator
        # (verified against Mathematica in test_compound_set_scoping_matches_mathematica).
        return Expr.__new__(cls, sympy.sympify(expr), limits)

    def _evaluate(self, **kwargs):
        expr, limits = self.args
        expanded = _expand_sum(expr, limits)
        if expanded is not None:
            return expanded
        if isinstance(limits, List) and len(limits.args) == 3:
            # Bounds still symbolic -- hand to sympy.Sum as before. NOTE this is
            # irreversible for the rules whose bound is an `Expon[...]` that later
            # resolves to a FRACTION: sympy.Sum leaves a fractional-bound sum
            # unevaluated forever and its iterator leaks (rule 9.3 #67, still open
            # -- RUBI_PORT_DEFECTS.md 50). Keeping the node deferred instead fixes
            # that, but only works if the node survives the matcher round trip,
            # and registering its head to achieve that regresses performance
            # badly (measured: one integral test file 2min -> >500s).
            i, imin, imax = limits.args
            return sympy.Sum(expr, (i, imin, imax)).doit()
        return sympy.Sum(expr, limits)


# Trick used to avoid name conflict with SymPy's Sum class:
Sum = SumWolfram


class Factorial(MathematicaExpr):
    """Mathematica ``Factorial[z]`` (``z!``)."""

    def __new__(cls, z):
        return Expr.__new__(cls, sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_Factorial(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.factorial(*self.args, evaluate=False)


class Zeta(MathematicaExpr):
    """Mathematica ``Zeta[s]`` / ``Zeta[s, a]`` — Riemann and Hurwitz zeta."""

    def __new__(cls, *args):
        return Expr.__new__(cls, *[sympy.sympify(a) for a in args])

    def _evaluate(self, **kwargs):
        return _eager.eager_Zeta(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.zeta(*self.args, evaluate=False)


class PolyGamma(MathematicaExpr):
    """Mathematica ``PolyGamma[z]`` / ``PolyGamma[n, z]``."""

    def __new__(cls, *args):
        return Expr.__new__(cls, *[sympy.sympify(a) for a in args])

    def _evaluate(self, **kwargs):
        return _eager.eager_PolyGamma(*self.args)

    def rewrite_as_standard_sympy(self):
        """Mathematica's one-argument ``PolyGamma[z]`` IS ``PolyGamma[0, z]``."""
        args = self.args if len(self.args) == 2 else (S.Zero,) + tuple(self.args)
        return sympy.polygamma(*args, evaluate=False)


class BesselJ(MathematicaExpr):
    """Mathematica ``BesselJ[n, z]`` — Bessel function of the first kind."""

    def __new__(cls, n, z):
        return Expr.__new__(cls, sympy.sympify(n), sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_BesselJ(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.besselj(*self.args, evaluate=False)


class ExpIntegralEi(MathematicaExpr):
    """Mathematica ``ExpIntegralEi[z]`` — the exponential integral Ei(z).

    https://reference.wolfram.com/language/ref/ExpIntegralEi.html

    A deferred node like every other Wolfram standard-library function here: this
    package is an INTERPRETER for the Wolfram language, so a head keeps its Wolfram
    identity and semantics even when SymPy happens to have the same function.
    ``doit()`` evaluates it to SymPy's ``Ei``.
    """

    def __new__(cls, z):
        return Expr.__new__(cls, sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_ExpIntegralEi(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.Ei(*self.args, evaluate=False)


class LogIntegral(MathematicaExpr):
    """Mathematica ``LogIntegral[z]`` — the logarithmic integral li(z).

    https://reference.wolfram.com/language/ref/LogIntegral.html

    ``doit()`` evaluates it to SymPy's ``li``; ``LogIntegral[1] == -Infinity`` in both.
    """

    def __new__(cls, z):
        return Expr.__new__(cls, sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_LogIntegral(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.li(*self.args, evaluate=False)


class Identity(MathematicaExpr):
    """Mathematica ``Identity[z]`` — returns its argument unchanged.

    Defined for interpreter completeness. The RULE generator does not emit it: unlike
    a real function, ``Identity`` carries no meaning of its own, so it is replaced by
    its argument at generation time (Rubi writes ``Identity[-1]*Int[u,x]`` only to stop
    the -1 folding away early).
    """

    def __new__(cls, z):
        return Expr.__new__(cls, sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_Identity(*self.args)

    def rewrite_as_standard_sympy(self):
        """``Identity[z]`` is just ``z`` -- there is no function left to keep."""
        return self.args[0]


class ExpIntegralE(MathematicaExpr):
    """Mathematica ``ExpIntegralE[n, z]`` — the exponential integral E_n(z)."""

    def __new__(cls, n, z):
        return Expr.__new__(cls, sympy.sympify(n), sympy.sympify(z))

    def _evaluate(self, **kwargs):
        return _eager.eager_ExpIntegralE(*self.args)

    def rewrite_as_standard_sympy(self):
        return sympy.expint(*self.args, evaluate=False)


class Root(MathematicaExpr):
    """Mathematica ``Root[poly, k]`` — the k-th root of *poly*, indexed from 1."""

    def __new__(cls, poly, k):
        return Expr.__new__(cls, sympy.sympify(poly), sympy.sympify(k))

    def _evaluate(self, **kwargs):
        return _eager.eager_Root(*self.args)


class Discriminant(MathematicaExpr):
    """Mathematica ``Discriminant[poly, x]`` — discriminant of *poly* with respect to *x*.

    Delegates to :func:`sympy_wolfram.functions_eager.eager_Discriminant`, which follows
    Mathematica on the degenerate cases SymPy handles differently (constant -> ``p^-2``,
    non-polynomial -> unevaluated).
    """

    def __new__(cls, p, x):
        return Expr.__new__(cls, sympy.sympify(p), sympy.sympify(x))

    def _evaluate(self, **kwargs):
        return _eager.eager_Discriminant(*self.args)


class Numerator(MathematicaExpr):
    """Mathematica Numerator[expr] -> numerator of rational expression.

    Delegates to the eager helper (which recurses through integer powers, pairing
    with :class:`Denominator`) rather than a bare ``sympy.numer``.
    """

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return _eager.eager_Numerator(self.args[0])


class Together(MathematicaExpr):
    """Mathematica Together[expr] -> combine fractions."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        expr, = self.args
        return sympy.together(expr)


class Expand(MathematicaExpr):
    """Mathematica ``Expand[expr]`` / ``Expand[expr, patt]`` -> multiply out.

    Deferred like :class:`Together`, so that in a rule it runs on the substituted
    expression rather than on the wildcards (Rubi: ``Int[Expand[u, x], x]``). Its
    standard-SymPy form is the eager ``expand(expr)``: ``Expand`` is an instruction
    to the CAS, and the value it denotes IS the expanded argument.

    The two-argument form restricts Mathematica's expansion to terms involving
    ``patt``; here it expands fully, which is equal as an expression and only
    differs in how the terms free of ``patt`` are laid out.
    """

    def __new__(cls, expr, *patt):
        if len(patt) > 1:
            raise TypeError('Expand takes 1 or 2 arguments, got %d' % (1 + len(patt)))
        return Expr.__new__(cls, sympy.sympify(expr), *[sympy.sympify(a) for a in patt])

    def _evaluate(self, **kwargs):
        return sympy.expand(self.args[0])

    def rewrite_as_standard_sympy(self):
        return sympy.expand(self.args[0])


class GCD(MathematicaExpr):
    """Mathematica GCD[a, b, ...] -> greatest common divisor."""

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        args = self.args
        if len(args) == 0:
            return S.Zero
        if len(args) == 1:
            return args[0]
        result = sympy.gcd(args[0], args[1])
        for a in args[2:]:
            result = sympy.gcd(result, a)
        return result


class Sign(MathematicaExpr):
    """Mathematica Sign[expr] -> sign (-1, 0, or 1)."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        expr, = self.args
        return sympy.sign(expr)


class Quotient(MathematicaExpr):
    """Mathematica Quotient[a, b] -> floor(a/b)."""

    def __new__(cls, a, b):
        return Expr.__new__(cls, a, b)

    def _evaluate(self, **kwargs):
        a, b = self.args
        return sympy.floor(a / b)


class EllipticPi(MathematicaExpr):
    """Mathematica EllipticPi[n, m] or EllipticPi[n, phi, m].

    Maps to sympy.elliptic_pi(n, m) or sympy.elliptic_pi(n, phi, m).
    """

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        if len(self.args) == 2:
            n, m = self.args
            return sympy.elliptic_pi(n, m)
        elif len(self.args) == 3:
            n, phi, m = self.args
            return sympy.elliptic_pi(n, phi, m)
        return self

    # Deliberately NO rewrite_as_standard_sympy: rubi_integrate keeps this node as a
    # deferred placeholder in its rules exactly because it does not self-translate
    # (its codegen checks the node's own __dict__). In standard-SymPy mode the
    # interpreter maps the head straight to sympy.elliptic_pi instead
    # (FFLConverter.STANDARD_SYMPY_FUNC_MAP), so the node never arises there.


class Apply(MathematicaExpr):
    """Mathematica Apply[f, {a, b, ...}] — apply f to list elements."""

    def __new__(cls, f, args):
        return Expr.__new__(cls, f, args)

    def _evaluate(self, **kwargs):
        f, args = self.args
        if hasattr(args, 'args'):
            return f(*args.args)
        return f(*args)


class FullSimplify(MathematicaExpr):
    """Mathematica FullSimplify[expr]."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return sympy.simplify(self.args[0])


class Simplify(MathematicaExpr):
    """Mathematica Simplify[expr]."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return sympy.simplify(self.args[0])


class FunctionExpand(MathematicaExpr):
    """Mathematica FunctionExpand[expr]."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return sympy.expand_func(self.args[0])


class Binomial(MathematicaExpr):
    """Mathematica Binomial[n, k]."""

    def __new__(cls, n, k):
        return Expr.__new__(cls, n, k)

    def _evaluate(self, **kwargs):
        return sympy.binomial(*self.args)


class ProductLog(MathematicaExpr):
    """Mathematica ``ProductLog[z]`` / ``ProductLog[k, z]`` — the Lambert W function.

    https://reference.wolfram.com/language/ref/ProductLog.html

    SymPy has this function, but the correspondence is NOT identity: Mathematica takes
    the branch index FIRST, SymPy takes it LAST, so ``ProductLog[k, z]`` is
    ``LambertW(z, k)``. Passing the arguments straight through would silently select
    the wrong branch. Verified on Mathematica 12.2:
    ``N[ProductLog[-1, -0.1]] == -3.577152063957297``.

    That mismatch is exactly why the head keeps its own node instead of being renamed
    away: the translation lives here, in one place, and ``doit()`` applies it.
    """

    def __new__(cls, *args):
        return Expr.__new__(cls, *[sympy.sympify(a) for a in args])

    def _evaluate(self, **kwargs):
        return _eager.eager_ProductLog(*self.args)

    def rewrite_as_standard_sympy(self):
        """The branch index MOVES: ``ProductLog[k, z]`` is ``LambertW(z, k)``."""
        if len(self.args) == 2:
            k, z = self.args
            return sympy.LambertW(z, k, evaluate=False)
        return sympy.LambertW(*self.args, evaluate=False)


class Floor(MathematicaExpr):
    """Mathematica Floor[x] or Floor[x, a].

    1-arg: Floor(x)    -> floor(x)
    2-arg: Floor(x, a) -> a * floor(x / a)   [rounds to nearest multiple of a]
    """

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        if len(self.args) == 1:
            return sympy.floor(self.args[0])
        elif len(self.args) == 2:
            x, a = self.args
            return a * sympy.floor(x / a)
        return self


class Hypergeometric2F1(MathematicaExpr):
    """Mathematica Hypergeometric2F1[a, b, c, z] -> hyper([a, b], [c], z)."""

    def __new__(cls, a, b, c, z):
        return Expr.__new__(cls, a, b, c, z)

    def _evaluate(self, **kwargs):
        a, b, c, z = self.args
        return sympy.hyper([a, b], [c], z)


# ---------------------------------------------------------------------------
# Bucket B1 — deferred nodes over the self-contained eager helpers.
# ---------------------------------------------------------------------------

class LeafCount(MathematicaExpr):
    """Mathematica LeafCount[expr] — count nodes in expression tree."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return Integer(_eager.eager_LeafCount(self.args[0]))


class Length(MathematicaExpr):
    """Mathematica Length[expr] — number of elements."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return Integer(_eager.eager_Length(self.args[0]))


class Not(MathematicaExpr):
    """Mathematica Not[expr] — logical negation."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return _eager.eager_Not(self.args[0])


class Complex(MathematicaExpr):
    """Mathematica Complex[re, im] — construct complex number re + I*im.

    Like the original in ``rubi_utils``, this is eager: ``__new__`` returns the
    SymPy value directly rather than a deferred node.
    """

    def __new__(cls, re, im):
        return _eager.eager_Complex(re, im)

    def _evaluate(self, **kwargs):
        return self


class Denominator(MathematicaExpr):
    """Mathematica Denominator[expr] — denominator of a rational expression."""

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        return _eager.eager_Denominator(*self.args)


class First(MathematicaExpr):
    """Mathematica First[expr] — first element."""

    def __new__(cls, expr, d=None):
        if d is None:
            return Expr.__new__(cls, expr)
        return Expr.__new__(cls, expr, d)

    def _evaluate(self, **kwargs):
        return _eager.eager_First(*self.args)


class Rest(MathematicaExpr):
    """Mathematica Rest[expr] — all elements but the first."""

    def __new__(cls, expr):
        return Expr.__new__(cls, expr)

    def _evaluate(self, **kwargs):
        return _eager.eager_Rest(self.args[0])


class Part(MathematicaExpr):
    """Mathematica Part[expr, n] — extract the n-th part (1-based)."""

    def __new__(cls, expr, *indices):
        return Expr.__new__(cls, expr, *indices)

    def _evaluate(self, **kwargs):
        return _eager.eager_Part(*self.args)


class Exponent(MathematicaExpr):
    """Mathematica Exponent[expr, form] / Exponent[expr, form, h]."""

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        return _eager.eager_Exponent(*self.args)


class Apart(MathematicaExpr):
    """Mathematica Apart[expr, x] — partial-fraction decomposition in x."""

    def __new__(cls, *args):
        safe = [sympy.sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        return _eager.eager_Apart(*self.args)

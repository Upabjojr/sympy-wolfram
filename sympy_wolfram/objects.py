# -*- coding: utf-8 -*-
"""Common Mathematica control/scoping constructs as SymPy expressions.

These classes model a small but useful subset of Mathematica semantics while
remaining fully symbolic until ``doit()`` is called, following SymPy
conventions for deferred evaluation.

All classes inherit from ``MathematicaExpr(Expr)`` and are fully symbolic
objects — they participate in SymPy expression trees and can be pattern-matched
or substituted into.  Evaluation is always explicit and on-demand via
``.doit()``.

Quick start
-----------
Local constants (simultaneous substitution)::

    >>> x, y = Symbol('x'), Symbol('y')
    >>> With(List(Set(x, Integer(2)), Set(y, Integer(3))), x**2 + y).doit()
    7

Lexical scoping with a computed local::

    >>> n = Symbol('n')
    >>> Module(List(Set(n, Integer(4))), n*(n + Integer(1))).doit()
    20

Conditional evaluation::

    >>> If(S.true, Integer(100), Integer(0)).doit()
    100

Iteration with value collection::

    >>> i = Symbol('i')
    >>> result = Reap(Do(Sow(i**2), List(i, Integer(4)))).doit()
    >>> list(result.args[1].args[0].args)
    [1, 4, 9, 16]

Non-local exit::

    >>> Catch(CompoundExpression(Throw(Integer(42)), Integer(0))).doit()
    42

Doctest configuration
---------------------
All examples below run with ``pytest --doctest-modules``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import sympy
from sympy import Add, Basic, Dummy, Expr, Integer, Mul, Pow, Rational, S, Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanFalse, BooleanTrue


Null = Symbol('Null')
"""Sentinel returned by procedural constructs that produce no value.

Equivalent to Mathematica's ``Null``.  Represented as ``Symbol('Null')`` so
it participates naturally in SymPy expression trees.

Examples
--------
>>> Do(Integer(0), List(Integer(0))).doit() == Null
True
>>> If(S.false, Integer(1)).doit() == Null
True
"""

_SOW_STACK: list[list[tuple[Any, Any]]] = []


class _ReturnSignal(Exception):
    """Internal signal raised by Return.doit() to unwind the call stack."""
    def __init__(self, value):
        self.value = value
        super().__init__(repr(value))


class _ThrowSignal(Exception):
    """Internal signal raised by Throw.doit() to unwind to an enclosing Catch."""
    def __init__(self, value, tag=None):
        self.value = value
        self.tag = tag
        super().__init__(repr((value, tag)))


# Bounded memo for MathematicaExpr.doit (see doit docstring note). Keyed by
# (class, node, deep); cleared wholesale when full.
_DOIT_CACHE: dict = {}
_DOIT_CACHE_MAX = 100000


class MathematicaExpr(Expr):
    """Abstract base class for all Mathematica-inspired SymPy expressions.

    Subclasses are fully symbolic SymPy ``Expr`` objects.  They are created
    unevaluated and evaluated on demand by calling ``.doit()``.

    Evaluation protocol
    -------------------
    ``doit(deep=True)`` (the default) first recursively calls ``.doit()`` on
    every argument, then calls ``_evaluate(**kwargs)`` on the resulting
    instance.  With ``deep=False`` the existing args are used as-is.

    Subclasses that need to control the evaluation order (e.g. ``With``,
    which must *not* evaluate the body before substituting local bindings)
    override ``doit()`` directly.

    Examples
    --------
    >>> isinstance(With(List(), Integer(1)), MathematicaExpr)
    True
    >>> isinstance(With(List(), Integer(1)), Expr)
    True
    >>> With(List(), Integer(1)).doit()
    1
    """

    # Every Wolfram node models a SCALAR value, so it commutes. Without this, sympy
    # computes ``is_commutative = None`` for the node and **False** for any Add/Mul
    # containing one -- and then refuses to distribute a numeric coefficient over that
    # Add. ``-(-1 - IntPart(m, 1))`` stayed as ``Mul(-1, Add(...))``, and Abs/signsimp
    # then flip-flopped between the two sign forms forever: a RecursionError that
    # aborted the whole integration (seen on ``Int[(c+d x)^4 Gamma[n, a+b x]]``; a
    # generic ``Function('f')(m, 1)``, which IS commutative, does not loop). It also
    # silently made every simplification treat Wolfram-node expressions as
    # noncommutative.
    is_commutative = True

    def doit(self, **kwargs):
        # MEMOISED (bounded): a deferred node's evaluation is a pure function of the
        # node (all _evaluate implementations delegate to eager utilities of the
        # args). During the integration DFS the SAME nodes are re-evaluated
        # constantly -- deferred-node doit held 15-18% of every profiled slow
        # integral. Only the plain call shape (no kwargs beyond `deep`) is cached;
        # scoping nodes (With/Module/If/...) override doit and bypass this. An
        # _evaluate that RAISES (e.g. Condition's StopIteration no-match protocol)
        # propagates before any cache store, so failures are never cached.
        deep = kwargs.get('deep', True)
        cache_key = None
        if not kwargs or set(kwargs) == {'deep'}:
            try:
                cache_key = (self.__class__, self, deep)
                hit = _DOIT_CACHE.get(cache_key)
                if hit is not None:
                    return hit
            except TypeError:        # unhashable arg somewhere -> uncached
                cache_key = None
        if deep:
            new_args = [
                arg.doit(**kwargs) if hasattr(arg, 'doit') else arg
                for arg in self.args
            ]
            instance = Expr.__new__(self.__class__, *new_args)
        else:
            instance = self
        result = instance._evaluate(**kwargs)
        # A node whose eager utility couldn't compute a value returns Python None.
        # Propagating None breaks the enclosing sympy operation (Add/Mul builds it
        # via sympify(None) -> SympifyError). Stay UNEVALUATED instead so the node
        # remains a legal expression; the DFS then treats the result as non-clean
        # and moves on rather than crashing.
        final = instance if result is None else result
        if cache_key is not None:
            if len(_DOIT_CACHE) >= _DOIT_CACHE_MAX:
                _DOIT_CACHE.clear()      # simple bounded reset (LRU not worth the cost)
            _DOIT_CACHE[cache_key] = final
        return final

    def _evaluate(self, **kwargs):
        raise NotImplementedError

    def rewrite_as_standard_sympy(self):
        """Re-express this node as the equivalent STANDARD SymPy function, UNEVALUATED.

        This is the bridge between the two languages, and it is deliberately distinct
        from :meth:`doit`:

        * ``doit()`` EVALUATES -- it applies Mathematica's semantics and computes.
          ``Factorial(5).doit()`` is ``120``.
        * ``rewrite_as_standard_sympy()`` TRANSLATES -- it swaps the Wolfram head for
          the SymPy one and stops. ``Factorial(5).rewrite_as_standard_sympy()`` is
          ``factorial(5)``, still a function application. That is what makes it usable
          on rule PATTERNS, whose arguments are wildcards that must survive intact.

        A head can be overloaded in Mathematica but not in SymPy, which is exactly why
        this is a METHOD on the node rather than a name table: ``Gamma[a]`` is the
        complete gamma function while ``Gamma[a, z]`` is the upper incomplete one, so
        the node inspects its own arity and picks.

        The default returns ``self``: most nodes (``With``, ``Module``, ``Condition``,
        ``Set``, ...) model Wolfram *language* constructs with no SymPy counterpart, and
        "there is no standard equivalent, I am already the best representation" is a
        meaningful answer rather than an error. Override it wherever a real equivalent
        exists.
        """
        return self


def rewrite_as_standard_sympy(expr):
    """Recursively rewrite every Wolfram node in *expr* to standard SymPy.

    Walks bottom-up so a nested node is translated before its parent is rebuilt, and
    leaves anything that is not a :class:`MathematicaExpr` untouched.
    """
    if isinstance(expr, (list, tuple)):
        return type(expr)(rewrite_as_standard_sympy(item) for item in expr)
    if not isinstance(expr, Basic):
        return expr
    args = getattr(expr, 'args', ())
    if args:
        new_args = [rewrite_as_standard_sympy(a) for a in args]
        if any(new is not old for new, old in zip(new_args, args)):
            try:
                expr = expr.func(*new_args)
            except (TypeError, ValueError):
                pass
    if isinstance(expr, MathematicaExpr):
        return expr.rewrite_as_standard_sympy()
    return expr


class Set(MathematicaExpr):
    """Mathematica ``Set[symbol, expr]`` — a name/value binding marker.

    ``Set`` is a *structural* node, not a reducible expression.  It is used
    exclusively inside ``List(...)`` passed to ``With``, ``Module``, or
    ``Block`` to associate a local name with its initial value.
    ``doit()`` returns ``self`` unchanged; the enclosing scoping construct
    is responsible for interpreting the binding.

    Parameters
    ----------
    symbol : sympy.Symbol
        The local variable name.
    expr : sympy.Expr
        The value to bind to *symbol*.

    Examples
    --------
    >>> x = Symbol('x')
    >>> b = Set(x, Integer(5))
    >>> b.args == (x, Integer(5))
    True
    >>> b.doit() == b          # structural marker, stays unevaluated
    True

    Used inside ``With``::

    >>> With(List(Set(x, Integer(5))), x + Integer(1)).doit()
    6
    """

    def __new__(cls, symbol: Any, expr: Any) -> "Set":
        return Expr.__new__(cls, sympify(symbol), sympify(expr))

    def _evaluate(self, **kwargs):
        return self


class SetDelayed(MathematicaExpr):
    """Mathematica ``SetDelayed[symbol, expr]`` (the ``:=`` operator).

    Like ``Set`` (``=``) it is a structural binding marker interpreted by the
    enclosing scope / sequence, but with DELAYED evaluation: the right-hand side is
    HELD and re-evaluated in the current environment every time the symbol is used,
    rather than evaluated once at assignment time. The observable difference from
    ``Set`` is when the RHS references a variable that is reassigned AFTER the
    binding (Mathematica-verified):

        Module[{u, y}, y = 2; u := y^2; y = 3; u]   (* SetDelayed -> 9 *)
        Module[{u, y}, y = 2; u  = y^2; y = 3; u]   (* Set        -> 4 *)

    ``CompoundExpression`` (and the scoping constructs) realise this by binding the
    symbol to the UNRESOLVED right-hand side and resolving bindings transitively at
    use time; ``Set`` resolves the value immediately. ``doit()`` returns ``self``.

    Parameters
    ----------
    symbol : sympy.Symbol
        The local variable name.
    expr : sympy.Expr
        The (held) value to bind to *symbol*.
    """

    def __new__(cls, symbol: Any, expr: Any) -> "SetDelayed":
        return Expr.__new__(cls, sympify(symbol), sympify(expr))

    def _evaluate(self, **kwargs):
        return self


class List(MathematicaExpr):
    """Mathematica ``List[e1, e2, …]`` — an ordered container of expressions.

    ``List`` is the primary container passed to scoping constructs
    (``With``, ``Module``, ``Block``) and to ``Do`` / ``Scan`` as the
    expression to iterate over.

    ``doit()`` returns ``self`` unchanged; ``List`` is a container, not a
    computation.  Items are accessible via ``args`` or iteration.

    Parameters
    ----------
    *items : sympy.Basic
        Zero or more SymPy-compatible elements.

    Examples
    --------
    >>> lst = List(Integer(1), Integer(2), Integer(3))
    >>> lst.args
    (1, 2, 3)
    >>> list(lst)          # supports iteration
    [1, 2, 3]
    >>> lst.doit() == lst  # container, not evaluated
    True
    >>> List().args        # empty list is allowed
    ()
    """

    def __new__(cls, *items: Any) -> "List":
        return Expr.__new__(cls, *[sympify(item) for item in items])

    def _evaluate(self, **kwargs):
        return self

    def __iter__(self) -> Iterator[Any]:
        yield from self.args


class CompoundExpression(MathematicaExpr):
    """Mathematica ``CompoundExpression[e1, e2, …]`` (semicolon operator).

    Evaluates each argument in order and returns the value of the last one.
    Equivalent to Mathematica's ``e1; e2; …; en``.  Intermediate values are
    discarded.  An empty ``CompoundExpression`` returns ``Null``.

    Parameters
    ----------
    *exprs : sympy.Basic
        Expressions to evaluate in sequence.

    Examples
    --------
    Basic sequential evaluation::

    >>> CompoundExpression(Integer(1), Integer(2), Integer(3)).doit()
    3

    Empty form returns ``Null``::

    >>> CompoundExpression().doit() == Null
    True

    Useful for grouping side effects inside ``Reap``::

    >>> result = Reap(CompoundExpression(Sow(Integer(10)), Sow(Integer(20)))).doit()
    >>> list(result.args[1].args[0].args)
    [10, 20]
    """

    def __new__(cls, *exprs: Any) -> "CompoundExpression":
        return Expr.__new__(cls, *[sympify(expr) for expr in exprs])

    def doit(self, **kwargs):
        result = Null
        # A ``Set``/``SetDelayed`` statement BINDS its symbol as a side effect for
        # every later statement in the sequence (Mathematica semantics). Rubi relies
        # on this, e.g. Module[{...,k,u}, u = Int[f(k)]; ... Sum[u, {k, 1, N}]] --
        # without propagating the u binding into the body, ``u`` (renamed to a Dummy
        # by the scoping construct) leaks unresolved into the Sum. Accumulate the
        # bindings and resolve them into the remaining statements.
        #
        #   * Set (=): fix the value NOW -- resolve the RHS against the current
        #     bindings at assignment time.
        #   * SetDelayed (:=): HOLD the RHS -- store it unresolved; it is resolved
        #     transitively at use time (`_resolve_bindings` iterates to a fixpoint),
        #     so a variable it references picks up the value in effect when used.
        #
        # SCOPING is correct because With/Module/Block rename their locals to fresh
        # Dummies at CONSTRUCTION (see _binding_substitutions): the bound symbol here
        # is that scope-unique Dummy, so resolution can never reach an identically-
        # named local of a nested scope (which owns a different Dummy). Verified
        # against Mathematica incl. nested/With shadowing -- see the tests.
        bindings = {}
        for expr in self.args:
            if isinstance(expr, (Set, SetDelayed)):
                var, val = expr.args
                if isinstance(expr, Set):
                    val = _resolve_bindings(val, bindings)   # eager: fix value now
                bindings[var] = val                          # SetDelayed holds RHS
                result = val
                continue
            result = _eval(_resolve_bindings(expr, bindings), **kwargs)
        return result

    def _evaluate(self, **kwargs):
        return self


class If(MathematicaExpr):
    """Mathematica ``If[condition, t]`` / ``If[condition, t, f]`` /
    ``If[condition, t, f, u]``.

    Evaluates *condition* and branches accordingly.

    * 2-arg form ``If[cond, t]``: returns *t* if *cond* is ``True``,
      otherwise ``Null``.
    * 3-arg form ``If[cond, t, f]``: returns *t* or *f* depending on *cond*.
    * 4-arg form ``If[cond, t, f, u]``: returns *u* when the condition
      is neither ``True`` nor ``False`` (symbolic / indeterminate).

    When the condition is symbolic and no 4-arg unknown branch is given,
    the expression remains unevaluated (returns a new ``If`` node with
    the evaluated condition).

    Parameters
    ----------
    condition : sympy.Basic
        Boolean test.  Compared against ``S.true`` / ``S.false``.
    true_value : sympy.Basic
        Result when *condition* is ``True``.
    false_value : sympy.Basic, optional
        Result when *condition* is ``False`` (default: ``Null``).
    unknown_value : sympy.Basic, optional
        Result when *condition* is neither ``True`` nor ``False``.

    Examples
    --------
    Two-argument form::

    >>> If(S.true, Integer(42)).doit()
    42
    >>> If(S.false, Integer(42)).doit() == Null
    True

    Three-argument form::

    >>> If(S.true, Integer(1), Integer(0)).doit()
    1
    >>> If(S.false, Integer(1), Integer(0)).doit()
    0

    Stays unevaluated when condition is symbolic (no unknown branch)::

    >>> x = Symbol('x')
    >>> result = If(sympy.Gt(x, Integer(0)), Integer(1), Integer(-1)).doit()
    >>> isinstance(result, If)
    True

    Four-argument form returns the unknown branch for symbolic conditions::

    >>> If(sympy.Gt(x, Integer(0)), Integer(1), Integer(-1), Integer(0)).doit()
    0

    Inside a ``With``, the condition resolves to a concrete boolean::

    >>> With(List(Set(x, Integer(5))),
    ...      If(sympy.Gt(x, Integer(3)), Integer(100), Integer(0))).doit()
    100
    """

    def __new__(cls, condition, true_value, false_value=None, unknown_value=None):
        args = [sympify(condition), sympify(true_value)]
        if false_value is not None:
            args.append(sympify(false_value))
        if unknown_value is not None:
            args.append(sympify(unknown_value))
        return Expr.__new__(cls, *args)

    def doit(self, **kwargs):
        cond = _eval(self.args[0], **kwargs)
        # A predicate used as an If CONDITION -- e.g. If[MatchQ[f, f1*Complex(0,j)], ...]
        # embedded in a rule's REPLACEMENT -- is a MathematicaConstraint, not a raw
        # boolean. In Wolfram, MatchQ[...] evaluates to True/False (resolving its LOCAL
        # pattern wildcards f1/j internally), so the If picks a branch. `_eval` alone
        # leaves it as the constraint object (a predicate's doit() returns self), so the
        # If would stay unevaluated and its MatchQ-local wildcards would leak into the
        # result. Evaluate the constraint here via .check(), exactly as Wolfram does.
        # Lazy import: constraints.py imports MathematicaExpr from this module.
        from sympy_wolfram.constraints import MathematicaConstraint
        if isinstance(cond, MathematicaConstraint):
            try:
                cond = S.true if cond.check(**kwargs) else S.false
            except Exception:
                pass
        if _is_true(cond):
            return _eval(self.args[1], **kwargs)
        if _is_false(cond):
            if len(self.args) >= 3:
                return _eval(self.args[2], **kwargs)
            return Null
        if len(self.args) >= 4:
            return _eval(self.args[3], **kwargs)
        return Expr.__new__(self.__class__, cond, *self.args[1:])

    def _evaluate(self, **kwargs):
        return self


def _bindings_list_from_dict(bindings: "Dict[Symbol, Optional[Any]]") -> "List":
    """Convert a ``{local: value}`` dict binding to the internal scoping ``List``.

    A ``None`` value denotes an UNINITIALISED local (`Module[{k}, ...]`) -- emitted
    as a bare symbol in the list rather than a ``Set``. This lets Module/With/Block
    accept the Python-dict form ``Module({r: v1, s: v2, k: None}, body)`` in addition
    to the Mathematica expression-tree form ``List(Set(r, v1), Set(s, v2), k)``.
    """
    items = []
    for k, v in bindings.items():
        items.append(k if v is None else Set(k, v))
    return List(*items)


class With(MathematicaExpr):
    """Mathematica ``With[{x=v, …}, body]`` — simultaneous local constants.

    ``With`` introduces local *constants*: all right-hand-side values are
    evaluated first (outside the body), then all variables are replaced
    simultaneously inside *body*, and finally the resulting expression is
    evaluated.  No variable can refer to another binding in the same ``With``.

    ``doit()`` is overridden to prevent *body* from being evaluated before
    substitution — the default ``MathematicaExpr.doit()`` would deep-evaluate
    all args first, which would reduce the body before local names are bound.

    If the body executes a ``Return[v]``, the value *v* is returned
    immediately and the remaining body is discarded.

    Parameters
    ----------
    bindings : List or dict
        Either the Mathematica expression-tree form -- a ``List`` of
        ``Set(symbol, value)`` nodes -- or the Python-dict form
        ``{symbol: value, ...}``.  In the dict form a ``None`` value denotes
        an uninitialised local (a bare symbol).  Both forms are normalised
        to a ``List`` at construction, so after construction every entry in
        ``self.args`` is a SymPy object.
    body : sympy.Expr
        The expression to evaluate after substitution.

    Examples
    --------
    Single binding::

    >>> x = Symbol('x')
    >>> With(List(Set(x, Integer(5))), x + Integer(1)).doit()
    6

    The dict form is equivalent to the ``List(Set(...))`` form::

    >>> With({x: Integer(5)}, x + Integer(1)).doit()
    6

    Multiple simultaneous bindings (order does not matter)::

    >>> x, y = Symbol('x'), Symbol('y')
    >>> With(List(Set(x, Integer(2)), Set(y, Integer(3))), x * y).doit()
    6

    Bindings are simultaneous — a later variable cannot reference an earlier one::

    >>> With(List(Set(x, Integer(4)), Set(y, x)), y).doit() == x
    True

    Nested ``With`` provides sequential binding::

    >>> With(List(Set(x, Integer(4))),
    ...      With(List(Set(y, x * Integer(2))), x + y)).doit()
    12

    Early exit via ``Return``::

    >>> With(List(Set(x, Integer(10))), Return(x * Integer(2))).doit()
    20
    """

    def __new__(cls, bindings: "Union[List, Dict[Symbol, Optional[Any]], list]",
                body: Any) -> "With":
        if isinstance(bindings, list):
            bindings = List(*bindings)
        if isinstance(bindings, dict):
            bindings = _bindings_list_from_dict(bindings)
        bindings, body = _bind_scope_locals(sympify(bindings), sympify(body))
        return Expr.__new__(cls, bindings, body)

    def doit(self, **kwargs):
        bindings, body = self.args
        subs = _binding_substitutions(bindings, evaluate_values=True, **kwargs)
        result = _substitute_body(body, subs)
        try:
            return _eval(result, **kwargs)
        except _ReturnSignal as signal:
            return signal.value

    def _evaluate(self, **kwargs):
        return self


class Module(MathematicaExpr):
    """Mathematica ``Module[{x, y=v, …}, body]`` — lexical scoping.

    ``Module`` creates a private lexical scope by replacing each local
    variable with a fresh uniquely-named symbol (``name$N`` where *N* is a
    monotone counter).  This prevents name clashes with outer variables of
    the same name.

    * A plain ``Symbol`` entry (e.g. ``x``) introduces a fresh uninitialized
      local.  The fresh symbol appears in the evaluated body.
    * A ``Set(x, v)`` entry initialises the fresh local to the value *v*.

    Like ``With``, ``Module`` catches ``Return`` signals.

    Parameters
    ----------
    locals_list : List or dict
        Either the Mathematica expression-tree form -- a ``List`` of
        ``Symbol`` or ``Set(symbol, init_value)`` entries -- or the
        Python-dict form ``{symbol: init_value, ...}``.  In the dict form a
        ``None`` value denotes an uninitialised local (equivalent to a plain
        ``Symbol`` entry).  Both forms are normalised to a ``List`` at
        construction, so after construction every entry in ``self.args`` is a
        SymPy object.
    body : sympy.Expr
        The expression to evaluate in the local scope.

    Examples
    --------
    Initialized local::

    >>> n = Symbol('n')
    >>> Module(List(Set(n, Integer(4))), n * (n + Integer(1))).doit()
    20

    The dict form accepts ``None`` for an uninitialised local::

    >>> k = Symbol('k')
    >>> isinstance(Module({k: None}, k).doit(), Symbol)
    True

    Uninitialized local produces a fresh symbol::

    >>> x = Symbol('x')
    >>> result = Module(List(x), x).doit()
    >>> isinstance(result, Symbol)
    True
    >>> isinstance(result, Symbol) and result != x    # Module creates a fresh symbol
    True

    Fresh symbols are independent across two ``Module`` calls::

    >>> r1 = Module(List(x), x).doit()
    >>> r2 = Module(List(x), x).doit()
    >>> r1 == r2
    False

    Fresh symbols are independent across two ``Module`` calls::

    >>> r1 = Module(List(x), x).doit()
    >>> r2 = Module(List(x), x).doit()
    >>> r1 == r2
    False

    Multiple locals with initializers::

    >>> a, b = Symbol('a'), Symbol('b')
    >>> Module(List(Set(a, Integer(3)), Set(b, Integer(4))), a**2 + b**2).doit()
    25
    """

    def __new__(cls, locals_list: "Union[List, Dict[Symbol, Optional[Any]], list]",
                body: Any) -> "Module":
        if isinstance(locals_list, list):
            locals_list = List(*locals_list)
        elif isinstance(locals_list, dict):
            locals_list = _bindings_list_from_dict(locals_list)
        locals_list, body = _bind_scope_locals(sympify(locals_list), sympify(body))
        return Expr.__new__(cls, locals_list, body)

    def doit(self, **kwargs):
        # The locals were already renamed to fresh Dummy symbols at construction
        # (see _bind_scope_locals), so this only has to bind the initialised ones;
        # an uninitialised local simply stays its Dummy in the result.
        locals_list, body = self.args
        subs = _binding_substitutions(locals_list, evaluate_values=True, **kwargs)
        result = _substitute_body(body, subs)
        try:
            return _eval(result, **kwargs)
        except _ReturnSignal as signal:
            return signal.value

    def _evaluate(self, **kwargs):
        return self


class Block(MathematicaExpr):
    """Mathematica ``Block[{x=v, …}, body]`` — dynamic variable scope.

    ``Block`` is similar to ``With`` but models Mathematica's *dynamic*
    scoping: local variables shadow outer bindings by name during the
    body's evaluation and are restored afterwards.  In this SymPy
    implementation, dynamic restoration is not required because SymPy
    expressions are immutable; ``Block`` therefore behaves like ``With``
    — it substitutes each binding into *body* and evaluates the result.

    The practical difference from ``With`` is semantic intent: ``With``
    documents that bindings are mathematical constants, while ``Block``
    documents that they are temporary variable overrides.

    Parameters
    ----------
    locals_list : List or dict
        Either the Mathematica expression-tree form -- a ``List`` of
        ``Set(symbol, value)`` bindings -- or the Python-dict form
        ``{symbol: value, ...}`` (a ``None`` value denotes an uninitialised
        local).  Both forms are normalised to a ``List`` at construction, so
        after construction every entry in ``self.args`` is a SymPy object.
    body : sympy.Expr
        The expression to evaluate under the local bindings.

    Examples
    --------
    Single binding::

    >>> x = Symbol('x')
    >>> Block(List(Set(x, Integer(10))), x + Integer(5)).doit()
    15

    The dict form is equivalent::

    >>> Block({x: Integer(10)}, x + Integer(5)).doit()
    15

    Multiple bindings::

    >>> a, b = Symbol('a'), Symbol('b')
    >>> Block(List(Set(a, Integer(3)), Set(b, Integer(7))), a * b).doit()
    21

    Nested expressions are evaluated after substitution::

    >>> Block(List(Set(x, Integer(2))), x**3 + x).doit()
    10
    """

    def __new__(cls, locals_list: "Union[List, Dict[Symbol, Optional[Any]], list]",
                body: Any) -> "Block":
        if isinstance(locals_list, list):
            locals_list = List(*locals_list)
        elif isinstance(locals_list, dict):
            locals_list = _bindings_list_from_dict(locals_list)
        locals_list, body = _bind_scope_locals(sympify(locals_list), sympify(body))
        return Expr.__new__(cls, locals_list, body)

    def doit(self, **kwargs):
        locals_list, body = self.args
        subs = _binding_substitutions(locals_list, evaluate_values=True, **kwargs)
        result = _substitute_body(body, subs)
        try:
            return _eval(result, **kwargs)
        except _ReturnSignal as signal:
            return signal.value

    def _evaluate(self, **kwargs):
        return self


class Return(MathematicaExpr):
    """Mathematica ``Return[value]`` — exit from a procedural construct.

    When ``doit()`` is called, ``Return`` raises an internal
    ``_ReturnSignal`` that unwinds the Python call stack until it reaches
    the nearest enclosing ``With``, ``Module``, or ``Block``
    (anything that catches ``_ReturnSignal``).  The caught value is then
    returned as the result of that construct.

    ``Return[]`` (no argument) returns ``Null``.

    Parameters
    ----------
    value : sympy.Basic, optional
        The value to return (default: ``Null``).

    Examples
    --------
    Early exit from ``With``::

    >>> x = Symbol('x')
    >>> With(List(Set(x, Integer(10))), Return(x * Integer(3))).doit()
    30

    ``Return[]`` returns ``Null``::

    >>> With(List(), Return()).doit() == Null
    True

    Exit from ``Module`` ignoring remaining body::

    >>> n = Symbol('n')
    >>> Module(
    ...     List(Set(n, Integer(5))),
    ...     CompoundExpression(Return(n**2), n + Integer(1))
    ... ).doit()
    25
    """

    def __new__(cls, value=Null):
        return Expr.__new__(cls, sympify(value))

    def doit(self, **kwargs):
        raise _ReturnSignal(_eval(self.args[0], **kwargs))

    def _evaluate(self, **kwargs):
        return self


class Throw(MathematicaExpr):
    """Mathematica ``Throw[value]`` / ``Throw[value, tag]``.

    ``Throw`` is the "raise" half of the ``Catch``/``Throw`` mechanism.
    Calling ``doit()`` raises a ``_ThrowSignal`` that propagates up the
    Python call stack until an enclosing ``Catch`` intercepts it.

    A *tag* can be attached to differentiate ``Throw`` sites; ``Catch``
    can then selectively match on the tag.

    Parameters
    ----------
    value : sympy.Basic
        The value to throw.
    tag : sympy.Basic, optional
        Tag attached to this throw for selective catching.

    Examples
    --------
    Simple throw caught by ``Catch``::

    >>> Catch(Throw(Integer(42))).doit()
    42

    Tagged throw::

    >>> tag = Symbol('myTag')
    >>> Catch(Throw(Integer(7), tag), tag).doit()
    7

    ``Throw`` inside a compound expression — remaining expressions are skipped::

    >>> Catch(CompoundExpression(Throw(Integer(1)), Integer(2))).doit()
    1
    """

    def __new__(cls, value, tag=None):
        args = [sympify(value)]
        if tag is not None:
            args.append(sympify(tag))
        return Expr.__new__(cls, *args)

    def doit(self, **kwargs):
        value = _eval(self.args[0], **kwargs)
        tag = _eval(self.args[1], **kwargs) if len(self.args) > 1 else None
        raise _ThrowSignal(value, tag)

    def _evaluate(self, **kwargs):
        return self


class Catch(MathematicaExpr):
    """Mathematica ``Catch[expr]`` / ``Catch[expr, tag]`` /
    ``Catch[expr, tag, handler]``.

    Evaluates *expr*; if a ``Throw`` occurs inside it, ``Catch``
    intercepts the thrown value and returns it.

    * 1-arg form: catches any ``Throw``, returns the thrown value.
    * 2-arg form: only catches ``Throw`` with a matching *tag*;
      unmatched throws propagate to the next outer ``Catch``.
    * 3-arg form: passes the thrown ``(value, tag)`` pair to a *handler*
      callable (e.g. a ``sympy.Lambda``) and returns its result.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to evaluate; may contain ``Throw``.
    form : sympy.Basic, optional
        Tag pattern; only throws whose tag equals *form* are caught.
    handler : callable or sympy.Lambda, optional
        Called as ``handler(value, tag)`` when a matching throw occurs.

    Examples
    --------
    No throw — normal evaluation::

    >>> Catch(Integer(5) + Integer(3)).doit()
    8

    Simple catch::

    >>> Catch(Throw(Integer(42))).doit()
    42

    Tagged catch — matching tag::

    >>> tag = Symbol('t')
    >>> Catch(Throw(Integer(99), tag), tag).doit()
    99

    Tagged catch — mismatched tag propagates::

    >>> import pytest
    >>> t1, t2 = Symbol('t1'), Symbol('t2')
    >>> with pytest.raises(Exception):
    ...     Catch(Throw(Integer(1), t1), t2).doit()

    Handler form — transform the caught value::

    >>> v = Symbol('v')
    >>> handler = sympy.Lambda(v, v * Integer(10))
    >>> Catch(Throw(Integer(3)), None, handler).doit()
    30
    """

    def __new__(cls, expr, form=None, handler=None):
        args = [sympify(expr)]
        if handler is not None:
            # Always store form at args[1] so handler is reliably at args[2].
            # Use Null as a sentinel meaning 'match any tag' when form is None.
            args.append(Null if form is None else sympify(form))
            args.append(sympify(handler))
        elif form is not None:
            args.append(sympify(form))
        return Expr.__new__(cls, *args)

    def doit(self, **kwargs):
        expr = self.args[0]
        if len(self.args) >= 3:
            form, handler = self.args[1], self.args[2]
        elif len(self.args) == 2:
            form, handler = self.args[1], None
        else:
            form = handler = None
        # Null sentinel means 'match any tag' (set by __new__ when form=None
        # but handler was provided).
        effective_form = None if (form is None or form == Null) else form
        try:
            return _eval(expr, **kwargs)
        except _ThrowSignal as signal:
            if effective_form is not None and signal.tag != effective_form:
                raise
            if handler is None:
                return signal.value
            return _apply_function(handler, signal.value, signal.tag)

    def _evaluate(self, **kwargs):
        return self


class Sow(MathematicaExpr):
    """Mathematica ``Sow[value]`` / ``Sow[value, tag]``.

    Deposits *value* into the nearest enclosing ``Reap`` collector and
    returns *value* as its own result.  If no ``Reap`` is active, the
    value is silently dropped and *value* is still returned.

    An optional *tag* can be used to associate the sown value with a
    label so that ``Reap`` can filter by tag.

    Parameters
    ----------
    value : sympy.Basic
        The value to collect.
    tag : sympy.Basic, optional
        A label attached to this sown value (default: ``Null``).

    Examples
    --------
    Returns the sown value regardless of whether a ``Reap`` is active::

    >>> Sow(Integer(7)).doit()
    7

    Inside ``Reap``, the value is collected::

    >>> result = Reap(Sow(Integer(7))).doit()
    >>> result.args[0]          # last evaluated value
    7
    >>> list(result.args[1].args[0].args)   # collected values
    [7]

    Tagged sow; only collected when ``Reap`` matches the same tag::

    >>> tag = Symbol('t')
    >>> result = Reap(CompoundExpression(Sow(Integer(1), tag),
    ...                                  Sow(Integer(2))), tag).doit()
    >>> list(result.args[1].args[0].args)   # only tagged value collected
    [1]
    """

    def __new__(cls, value, tag=None):
        args = [sympify(value)]
        if tag is not None:
            args.append(sympify(tag))
        return Expr.__new__(cls, *args)

    def doit(self, **kwargs):
        value = _eval(self.args[0], **kwargs)
        tag = _eval(self.args[1], **kwargs) if len(self.args) > 1 else Null
        if _SOW_STACK:
            _SOW_STACK[-1].append((tag, value))
        return value

    def _evaluate(self, **kwargs):
        return self


class Reap(MathematicaExpr):
    """Mathematica ``Reap[expr]`` / ``Reap[expr, tag]``.

    Evaluates *expr* while collecting all values deposited by ``Sow``
    calls inside it.  Returns a two-element ``List``:

    ``List(last_value, List(List(sown_1, sown_2, …)))``,

    mirroring Mathematica's ``{lastValue, {{sown1, sown2, …}}}``.  The
    inner double-wrapping is intentional: the outer ``List`` holds one
    sub-list per distinct tag when a tag pattern is used.

    An optional *tag* filters which sown values are collected;
    only values sown with a matching tag appear in the output.

    Parameters
    ----------
    expr : sympy.Basic
        Expression to evaluate; may contain ``Sow`` calls.
    pattern : sympy.Basic, optional
        If given, only collect values whose tag equals *pattern*.

    Examples
    --------
    Collect multiple values::

    >>> i = Symbol('i')
    >>> result = Reap(Do(Sow(i), List(i, Integer(4)))).doit()
    >>> result.args[0] == Null      # Do returns Null
    True
    >>> list(result.args[1].args[0].args)
    [1, 2, 3, 4]

    Combine with ``CompoundExpression``::

    >>> result = Reap(CompoundExpression(Sow(Integer(1)),
    ...                                  Sow(Integer(2)))).doit()
    >>> result.args[0]              # last evaluated value (Integer(2))
    2
    >>> list(result.args[1].args[0].args)
    [1, 2]

    Filtered by tag::

    >>> tag = Symbol('t')
    >>> result = Reap(CompoundExpression(Sow(Integer(10), tag),
    ...                                  Sow(Integer(20))), tag).doit()
    >>> list(result.args[1].args[0].args)
    [10]

    Nested ``Reap`` calls are independent::

    >>> outer = Reap(CompoundExpression(
    ...     Sow(Integer(1)),
    ...     Reap(Sow(Integer(2))))).doit()
    >>> list(outer.args[1].args[0].args)  # outer only sees value 1
    [1]
    """

    def __new__(cls, expr, pattern=None):
        args = [sympify(expr)]
        if pattern is not None:
            args.append(sympify(pattern))
        return Expr.__new__(cls, *args)

    def doit(self, **kwargs):
        expr = self.args[0]
        pattern = self.args[1] if len(self.args) > 1 else None
        bag: list[tuple[Any, Any]] = []
        _SOW_STACK.append(bag)
        try:
            result = _eval(expr, **kwargs)
        finally:
            _SOW_STACK.pop()
        values = [value for tag, value in bag if pattern is None or tag == pattern]
        return List(result, List(List(*values)))

    def _evaluate(self, **kwargs):
        return self


class Scan(MathematicaExpr):
    """Mathematica ``Scan[f, list]`` — apply *f* to each element for side effects.

    Applies *function* to every element of *expr* in order.  Return values
    are discarded.  Always returns ``Null``.

    *function* may be a ``sympy.Lambda``, a ``sympy.Function`` subclass
    instance, or any Python callable.

    If *function* triggers a ``Return`` signal during evaluation, ``Scan``
    catches it and immediately returns the signalled value (matching
    Mathematica's ``Return`` semantics inside ``Scan``).

    Parameters
    ----------
    function : callable or sympy.Lambda
        The function to apply to each element.
    expr : List
        The list of elements to scan over.

    Examples
    --------
    Returns ``Null`` after applying the function::

    >>> f = Symbol('f')
    >>> Scan(f, List(Integer(1), Integer(2), Integer(3))).doit() == Null
    True

    Collect squares via ``Sow`` inside a ``Reap``::

    >>> x = Symbol('x')
    >>> result = Reap(
    ...     Scan(sympy.Lambda(x, Sow(x**2)),
    ...          List(Integer(1), Integer(2), Integer(3)))
    ... ).doit()
    >>> list(result.args[1].args[0].args)
    [1, 4, 9]

    Early exit with ``Return``::

    >>> sentinel = Symbol('done')
    >>> Scan(
    ...     sympy.Lambda(x, If(sympy.Eq(x, Integer(2)),
    ...                        Return(sentinel), Sow(x))),
    ...     List(Integer(1), Integer(2), Integer(3))
    ... ).doit() == sentinel
    True
    """

    def __new__(cls, function, expr):
        return Expr.__new__(cls, sympify(function), sympify(expr))

    def doit(self, **kwargs):
        function, expr = self.args
        for item in _scan_items(expr):
            try:
                _eval(_apply_function(function, item), **kwargs)
            except _ReturnSignal as signal:
                return signal.value
        return Null

    def _evaluate(self, **kwargs):
        return self


class Do(MathematicaExpr):
    """Mathematica ``Do[body, spec]`` — imperative iteration.

    Executes *body* repeatedly according to one or more iterator
    specifications.  Always returns ``Null`` (use ``Sow``/``Reap`` or
    ``Scan`` to collect values).

    Iterator specification forms (passed as a ``List``):

    * ``List(n)``            — repeat *body* exactly *n* times.
    * ``List(i, n)``         — *i* runs from 1 to *n* (inclusive), step 1.
    * ``List(i, imin, imax)`` — *i* runs from *imin* to *imax*, step 1.
    * ``List(i, imin, imax, step)`` — *i* runs from *imin* to *imax* by
      *step* (may be negative for descending ranges).

    Multiple iterator specs produce nested loops (first spec is the
    outermost loop).

    If *body* triggers a ``Return`` signal, ``Do`` catches it and
    immediately returns the signalled value.

    Parameters
    ----------
    expr : sympy.Basic
        The body expression to evaluate on each iteration.
    *iter_specs : List
        One or more ``List`` iterator specifications (see forms above).

    Examples
    --------
    Fixed repetition::

    >>> Do(Integer(0), List(Integer(5))).doit() == Null
    True

    With iterator variable (1 to *n*, implicit)::

    >>> i = Symbol('i')
    >>> result = Reap(Do(Sow(i), List(i, Integer(4)))).doit()
    >>> list(result.args[1].args[0].args)
    [1, 2, 3, 4]

    Explicit range::

    >>> result = Reap(Do(Sow(i), List(i, Integer(3), Integer(7)))).doit()
    >>> list(result.args[1].args[0].args)
    [3, 4, 5, 6, 7]

    With step::

    >>> result = Reap(Do(Sow(i), List(i, Integer(1), Integer(9), Integer(2)))).doit()
    >>> list(result.args[1].args[0].args)
    [1, 3, 5, 7, 9]

    Descending step::

    >>> result = Reap(Do(Sow(i), List(i, Integer(5), Integer(1), Integer(-1)))).doit()
    >>> list(result.args[1].args[0].args)
    [5, 4, 3, 2, 1]

    Nested loops (matrix traversal)::

    >>> j = Symbol('j')
    >>> result = Reap(Do(Sow(i * Integer(10) + j),
    ...                  List(i, Integer(2)),
    ...                  List(j, Integer(2)))).doit()
    >>> list(result.args[1].args[0].args)
    [11, 12, 21, 22]
    """

    def __new__(cls, expr, *iter_specs):
        return Expr.__new__(cls, sympify(expr), *[sympify(spec) for spec in iter_specs])

    def doit(self, **kwargs):
        body = self.args[0]
        iter_specs = self.args[1:]
        try:
            _execute_do(body, iter_specs, kwargs)
        except _ReturnSignal as signal:
            return signal.value
        return Null

    def _evaluate(self, **kwargs):
        return self


class Head(MathematicaExpr):
    """Mathematica ``Head[expr]`` — the top-level constructor of an expression.

    Returns a ``Symbol`` whose name is the Mathematica-style head of the
    expression.  The mapping follows standard Wolfram conventions:

    ============================================ ==============
    Expression type                              Head returned
    ============================================ ==============
    ``Integer``                                  ``Integer``
    ``Rational`` (non-integer)                   ``Rational``
    ``Symbol``                                   ``Symbol``
    ``Add`` (sum)                                ``Plus``
    ``Mul`` (product)                            ``Times``
    ``Pow`` (power)                              ``Power``
    ``List``                                     ``List``
    ``sympy.Function`` subclass (e.g. ``sin``)   function name
    ``MathematicaExpr`` subclass                 class name
    ``S.true`` / ``S.false``                     ``True`` / ``False``
    ============================================ ==============

    Examples
    --------
    Numeric types::

    >>> Head(Integer(5)).doit()
    Integer
    >>> Head(Rational(2, 3)).doit()
    Rational

    Symbols and lists::

    >>> x = Symbol('x')
    >>> Head(x).doit()
    Symbol
    >>> Head(List(Integer(1), Integer(2))).doit()
    List

    Arithmetic operations::

    >>> a, b = Symbol('a'), Symbol('b')
    >>> Head(a + b).doit()
    Plus
    >>> Head(a * b).doit()
    Times
    >>> Head(a**b).doit()
    Power

    SymPy functions::

    >>> Head(sympy.sin(x)).doit()
    sin
    >>> Head(sympy.exp(x)).doit()
    exp

    Boolean constants::

    >>> Head(S.true).doit()
    True
    """

    def __new__(cls, expr):
        return Expr.__new__(cls, sympify(expr))

    def _evaluate(self, **kwargs):
        expr = self.args[0]
        return Symbol(_head_name(expr))


class D(MathematicaExpr):
    """Derivative"""
    def __new__(cls, f, x):
        obj = MathematicaExpr.__new__(cls, f, x)
        return obj

    def _evaluate(self, **kwargs):
        f, x = self.args
        return sympy.diff(f, x)


def _eval(expr, **kwargs):
    return expr.doit(**kwargs) if hasattr(expr, 'doit') else expr


def _is_true(value) -> bool:
    return value is True or value == S.true or isinstance(value, BooleanTrue)


def _is_false(value) -> bool:
    return value is False or value == S.false or isinstance(value, BooleanFalse)


def _list_items(expr) -> Iterable[Any]:
    if isinstance(expr, List):
        return expr.args
    if isinstance(expr, (tuple, list)):
        return expr
    return (expr,)


def _bind_scope_locals(bindings: Basic, body: Any) -> "Tuple[List, Any]":
    """Close a scoping construct over its own locals, by alpha-renaming them to
    ``Dummy`` symbols.

    Called from ``With``/``Module``/``Block`` ``__new__``, so a scoping node is
    ALREADY closed over its locals the moment it exists. That is what makes the
    scope well-defined: a ``Dummy`` is unique by identity, so a later
    ``expr.subs(Symbol('a'), ...)`` from outside cannot reach a local named ``a``,
    and no external "rename the locals first" pass is needed.

    Two details matter:

    * a binding's VALUE is evaluated in the ENCLOSING scope, so it is deliberately
      NOT renamed -- ``With[{a = f[a]}, ...]`` binds the local ``a`` to the *outer*
      ``a``, exactly as Mathematica does;
    * locals that are already ``Dummy`` are left alone, which makes this idempotent.
      SymPy rebuilds expressions constantly (``expr.func(*expr.args)``), and
      re-binding on every rebuild would mint fresh dummies each time and detach the
      binder from its body.

    Nesting falls out for free: an inner scope is constructed before the outer one,
    so its body already refers to its own dummies and the outer rename cannot touch
    them -- inner bindings shadow outer ones correctly.
    """
    renaming = {}
    new_items = []
    for item in _list_items(bindings):
        if isinstance(item, Set) and isinstance(item.args[0], Symbol):
            local, value = item.args
            if isinstance(local, Dummy):
                new_items.append(item)          # already bound
            else:
                fresh = Dummy(local.name)
                renaming[local] = fresh
                new_items.append(Set(fresh, value))   # value keeps the outer scope
        elif isinstance(item, Symbol) and not isinstance(item, Dummy):
            fresh = Dummy(item.name)
            renaming[item] = fresh
            new_items.append(fresh)
        else:
            new_items.append(item)
    if renaming and isinstance(body, Basic):
        body = body.xreplace(renaming)
    return List(*new_items), body


def _binding_substitutions(bindings: Basic, evaluate_values: bool, **kwargs) -> "Dict[Basic, Any]":
    subs = {}
    for item in _list_items(bindings):
        if isinstance(item, Set):
            symbol, value = item.args
            if evaluate_values:
                evaluated = _eval(value, **kwargs)
                # A utility can evaluate to Python None (it couldn't compute the
                # binding for this input). Substituting None would raise
                # SympifyError in xreplace; keep the raw (still-valid, possibly
                # deferred) value instead so the binding stays a legal expression --
                # the rule then simply yields a non-clean result and the DFS moves on.
                if evaluated is None:
                    evaluated = value
                elif isinstance(evaluated, (list, tuple)):
                    # Mathematica utilities that "return a list" return a List
                    # EXPRESSION, and Rubi's rules then read it with Part. Several of
                    # our eager helpers (FunctionOfLinear, FunctionOfLog, ...) hand
                    # back a PYTHON list instead; substituting that into the body
                    # explodes inside SymPy's xreplace, which requires every value to
                    # be a Basic. Wrap it here, at the one place bindings are made,
                    # rather than in each helper.
                    evaluated = List(*evaluated)
                subs[symbol] = evaluated
            else:
                subs[symbol] = value
    return subs


def _is_condition_wrapper(expr) -> bool:
    return getattr(expr.__class__, '__name__', '') == 'Condition' and len(getattr(expr, 'args', ())) == 2


def _substitute_body(body: Basic, substitutions: "Dict[Basic, Any]") -> Basic:
    if _is_condition_wrapper(body):
        expr, test = body.args
        return body.func(expr.xreplace(substitutions), test.xreplace(substitutions))
    return body.xreplace(substitutions)


def _resolve_bindings(expr: Any, bindings: "Dict[Basic, Any]", _max_iter: int = 64) -> Any:
    """Substitute ``bindings`` into ``expr`` to a FIXPOINT.

    A single ``xreplace`` only substitutes one level; iterating to a fixpoint is what
    makes ``SetDelayed`` transitive -- e.g. with ``{u: y**2, y: 3}`` resolving ``u``
    gives ``y**2`` then ``9``. The iteration is BOUNDED: a self-referential delayed
    binding (``u := u + 100``; which Mathematica reports as an unterminating recursion
    and no Rubi rule produces) stops after ``_max_iter`` rounds instead of looping
    forever. For ``Set`` bindings, whose values are already resolved, this converges
    in one round.
    """
    if not bindings or not isinstance(expr, Basic):
        return expr
    for _ in range(_max_iter):
        new = expr.xreplace(bindings)
        if new == expr:
            return new
        expr = new
    return expr


def _condition_holds(test, **kwargs) -> bool:
    # Evaluate boolean connectives LAZILY, conjunct-by-conjunct, BEFORE evaluating
    # the whole test. Mathematica's And/Or/Not are short-circuiting; more
    # importantly, evaluating a combined And upfront (via _eval) can build/sort a
    # structure that embeds a non-Expr sentinel (a utility function returning the
    # symbol False, e.g. DerivativeDivides), which sympy's canonical sort chokes on
    # ('bool' object has no attribute 'is_Float'). Evaluating each operand in
    # isolation avoids that and lets a False conjunct short-circuit cleanly.
    if isinstance(test, sympy.logic.boolalg.Not):
        return not _condition_holds(test.args[0], **kwargs)
    if isinstance(test, sympy.logic.boolalg.And):
        return all(_condition_holds(arg, **kwargs) for arg in test.args)
    if isinstance(test, sympy.logic.boolalg.Or):
        return any(_condition_holds(arg, **kwargs) for arg in test.args)
    test = _eval(test, **kwargs) if hasattr(test, 'doit') else test
    if _is_true(test):
        return True
    if _is_false(test):
        return False
    if hasattr(test, 'check') and callable(test.check):
        return bool(test.check(**kwargs))
    return False


class Condition(MathematicaExpr):
    """Mathematica ``Condition[expr, test]`` (``expr /; test``).

    Evaluates the *test* first and returns ``expr`` (evaluated) only when the test
    holds; a failing test raises ``StopIteration`` (the "no pattern match" signal).

    The default deep ``doit`` would evaluate ``expr`` (an argument) BEFORE the test,
    which is wrong — the body may be invalid when the test fails, and Mathematica's
    ``Set[sym, value]`` assignments inside the test bind locals the body then uses
    (e.g. ``Module[{q,r}, q*r*… /; (Set[q,…] =!= False && Set[r,…] =!= False)]``).
    Since these expressions are immutable, we capture each ``Set``'s value while
    evaluating the test and substitute those bindings into the body when it holds.
    """

    def __new__(cls, expr, test):
        return Expr.__new__(cls, expr, test)

    def doit(self, **kwargs):
        return self._eval_condition(**kwargs)

    def _evaluate(self, **kwargs):
        return self._eval_condition(**kwargs)

    def _eval_condition(self, **kwargs):
        expr, test = self.args
        bindings = {}

        def _eval_set(node):
            sym, val = node.args
            value = val.doit(**kwargs) if hasattr(val, 'doit') else val
            bindings[sym] = value
            return value

        test_eval = test
        if isinstance(test, sympy.Basic):
            test_eval = test.replace(lambda n: isinstance(n, Set), _eval_set)
        if _condition_holds(test_eval, **kwargs):
            body = expr.subs(bindings) if bindings else expr
            return body.doit(**kwargs) if hasattr(body, 'doit') else body
        raise StopIteration


def _apply_function(function, *items):
    if isinstance(function, sympy.Lambda):
        # Slice args to Lambda's declared arity: a 1-variable Lambda(v, expr)
        # still works when called as handler(value, tag).
        n = len(function.variables)
        return function(*items[:n])
    if isinstance(function, Expr) and getattr(function, 'is_Function', False):
        return function.func(*items)
    if callable(function):
        return function(*items)
    return sympy.Function(str(function))(*items)


def _scan_items(expr):
    if isinstance(expr, List):
        return expr.args
    if isinstance(expr, (tuple, list)):
        return expr
    return (expr,)


def _execute_do(body, iter_specs, kwargs):
    if not iter_specs:
        _eval(body, **kwargs)
        return
    current, *rest = iter_specs
    for subs in _iterator_substitutions(current, **kwargs):
        next_body = body.xreplace(subs)
        _execute_do(next_body, rest, kwargs)


def _iterator_substitutions(spec, **kwargs):
    if not isinstance(spec, List):
        raise TypeError('Do iterator specification must be a List expression')
    items = list(spec.args)
    if not items:
        return []

    if isinstance(items[0], Symbol):
        var = items[0]
        if len(items) == 2:
            imin = Integer(1)
            imax = _eval(items[1], **kwargs)
            step = Integer(1)
        elif len(items) == 3:
            imin = _eval(items[1], **kwargs)
            imax = _eval(items[2], **kwargs)
            step = Integer(1)
        elif len(items) == 4:
            imin = _eval(items[1], **kwargs)
            imax = _eval(items[2], **kwargs)
            step = _eval(items[3], **kwargs)
        else:
            raise ValueError('Unsupported Do iterator specification')
        return ({var: value} for value in _inclusive_range(imin, imax, step))

    if len(items) == 1:
        n = _eval(items[0], **kwargs)
        return ({} for _ in _inclusive_range(Integer(1), n, Integer(1)))

    raise ValueError('Unsupported Do iterator specification')


def _inclusive_range(imin, imax, step):
    start = int(imin)
    stop = int(imax)
    delta = int(step)
    if delta == 0:
        raise ValueError('Do step must be non-zero')
    if delta > 0:
        return [Integer(i) for i in range(start, stop + 1, delta)]
    return [Integer(i) for i in range(start, stop - 1, delta)]


def _head_name(expr) -> str:
    if expr == S.true:
        return 'True'
    if expr == S.false:
        return 'False'
    if expr == Null:
        return 'Symbol'
    if isinstance(expr, Integer):
        return 'Integer'
    if isinstance(expr, Rational) and not isinstance(expr, Integer):
        return 'Rational'
    if isinstance(expr, Symbol):
        return 'Symbol'
    if isinstance(expr, List):
        return 'List'
    if isinstance(expr, Add):
        return 'Plus'
    if isinstance(expr, Mul):
        return 'Times'
    if isinstance(expr, Pow):
        return 'Power'
    if isinstance(expr, sympy.Function):
        return expr.func.__name__
    if isinstance(expr, MathematicaExpr):
        return expr.__class__.__name__
    return expr.func.__name__


__all__ = [
    'Block',
    'rewrite_as_standard_sympy',
    'Catch',
    'CompoundExpression',
    'Condition',
    'D',
    'Gamma',
    'Do',
    'Head',
    'If',
    'List',
    'MathematicaExpr',
    'Module',
    'Null',
    'Reap',
    'Return',
    'Scan',
    'Set',
    'SetDelayed',
    'Sow',
    'Throw',
    'With',
]


# =============================================================================
# Gamma
# =============================================================================

class Gamma(MathematicaExpr):
    """Mathematica Gamma[z] or Gamma[a, z] -> sympy.gamma / sympy.uppergamma."""

    def __new__(cls, *args):
        safe = [sympify(a) for a in args]
        return Expr.__new__(cls, *safe)

    def _evaluate(self, **kwargs):
        if len(self.args) == 1:
            return sympy.gamma(self.args[0])
        elif len(self.args) == 2:
            return sympy.uppergamma(self.args[0], self.args[1])
        else:
            raise NotImplementedError

    def rewrite_as_standard_sympy(self):
        """Mathematica OVERLOADS Gamma by arity; SymPy uses two different functions.

        ``Gamma[a]``    -> ``gamma(a)``          (complete)
        ``Gamma[a, z]`` -> ``uppergamma(a, z)``  (UPPER incomplete)

        This is the case that motivated the whole protocol: no name table can express
        it, because which SymPy function is correct depends on how many arguments the
        node has.
        """
        if len(self.args) == 1:
            return sympy.gamma(*self.args, evaluate=False)
        if len(self.args) == 2:
            return sympy.uppergamma(*self.args, evaluate=False)
        return self

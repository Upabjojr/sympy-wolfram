# Wolfram nodes and the three kinds of evaluation

Mathematica functions that have no SymPy twin — or whose SymPy twin behaves
differently — are modelled here as **deferred nodes** plus, where useful, **eager
functions**. This document explains the split, the three evaluation operations
(`doit`, `rewrite_as_standard_sympy`, eager calls), and the constraint classes built
on top.

Every example is a doctest, executed by `sympy_wolfram/tests/test_docs.py`.

```python
>>> from sympy import Symbol, Integer
>>> x, a, b = Symbol('x'), Symbol('a'), Symbol('b')

```

---

## 1. Deferred nodes: `MathematicaExpr`

A deferred node is a SymPy `Expr` subclass that stays **inert** when constructed — it
holds its arguments like an undefined function, so it can sit inside rule patterns and
replacements without evaluating prematurely:

```python
>>> from sympy_wolfram.mathematica_functions import Coefficient
>>> c = Coefficient(a*x**2 + b*x, x, 2)
>>> c
Coefficient(a*x**2 + b*x, x, 2)

```

Being a real `Expr`, it composes with SymPy arithmetic, substitution and traversal —
which is precisely what rule replacements need before their wildcards are bound.

### `doit()` — evaluate with Wolfram semantics

`doit()` performs the actual computation, following Mathematica's behaviour for that
head (each node implements `_evaluate`):

```python
>>> c.doit()
a

```

The inert/`doit` split is what lets a rule's replacement mention `Coefficient(...)`
over wildcards: the node is built symbolically at rule-definition time and evaluated
at fire time, when the wildcards have values.

### `rewrite_as_standard_sympy()` — translate the head, do not evaluate

Some Wolfram heads have a standard-SymPy counterpart under a different name or
argument order. `rewrite_as_standard_sympy()` converts the *head only* — no
computation:

```python
>>> from sympy_wolfram.mathematica_functions import ExpIntegralEi
>>> node = ExpIntegralEi(x)
>>> node.rewrite_as_standard_sympy()
Ei(x)

```

A module-level helper walks a whole expression and rewrites every node that supports
the protocol:

```python
>>> from sympy_wolfram.objects import rewrite_as_standard_sympy
>>> rewrite_as_standard_sympy(ExpIntegralEi(x) + 1)
Ei(x) + 1

```

The distinction matters: `doit` answers "what is the value", `rewrite_as_standard_sympy`
answers "what is this called in plain SymPy". Final results handed to SymPy users go
through the rewrite; intermediate rule evaluation goes through `doit`.

> **Translation traps** (each found the hard way, all cross-checked numerically
> against Mathematica): argument order can differ (`ProductLog[k, z]` is
> `LambertW(z, k)`), arity can differ (`PolyGamma[z]` is `polygamma(0, z)`), and
> same-family names can point at different conventions (`Gamma[a, z]` is the *upper*
> incomplete gamma). Structural review is not enough for a new head translation — only
> numeric comparison catches these.

## 2. Eager functions: `eager_<Name>`

Predicates and helpers that rules call *immediately* (mostly in constraints) are plain
Python functions. Naming convention: when a deferred node of the same Wolfram name
exists, the eager function is prefixed `eager_`, and the bare name stays with the
node class.

```python
>>> from sympy_wolfram.functions_eager import eager_FreeQ
>>> eager_FreeQ(a + b, x)       # FreeQ[a+b, x]
True
>>> eager_FreeQ(a + x, x)
False
>>> eager_FreeQ([a, b], x)      # FreeQ[{a, b}, x]: every element must be free
True

```

Eager functions implement Mathematica's semantics, not SymPy's, wherever the two
disagree — e.g. Wolfram `IntegerPart` truncates toward zero where a naive `floor`
would not. When porting one, the reference is what Mathematica actually returns.

## 3. Constraints: `MathematicaConstraint`

Rule guards live one level up from eager functions: a `MathematicaConstraint` (built
on `sympy_matching`'s `SymPyMatchingConstraint`) is a first-class SymPy Boolean that
resolves its arguments from the matched wildcards **by name**, then delegates to the
eager predicate:

```python
>>> from sympy_wolfram.constraints_wolfram import FreeQ
>>> from sympy_matching.wild import WildSymbol
>>> a_, b_ = WildSymbol('a'), WildSymbol('b')
>>> FreeQ(a_, x).check(a=b)         # the binding a -> b is free of x
True
>>> FreeQ(a_, x).check(a=x**2)
False
>>> FreeQ([a_, b_], x).check(a=b, b=Integer(3))     # FreeQ[{a,b}, x]
True
>>> FreeQ([a_, b_], x).check(a=b, b=x)
False

```

Being SymPy Booleans, constraints compose with `Not`/`And`/`Or` and print readably in
rule listings:

```python
>>> FreeQ([a_, b_], x)
FreeQ([a, b], x)

```

### Relation to `omnimatch`'s `FreeOf`

`FreeOf` (in omnimatch core) is the same predicate at the OmniMatch-constraint level, and
deliberately accepts the same argument shapes — one variable or a list, names as
strings or as objects. `FreeQ` is the Wolfram-named, SymPy-Boolean-valued layer over
the idea; use it in `SymPyReplacementPattern` guards, and `FreeOf` when attaching
constraints directly to a OmniMatch `Pattern`.

## 4. Choosing the right tool

| you need | use |
|---|---|
| a Wolfram function inside a rule pattern/replacement | the **deferred node** |
| its value at rule-fire time | the node's **`doit()`** (the machinery calls it) |
| the plain-SymPy spelling of a result | **`rewrite_as_standard_sympy`** |
| a predicate inside a constraint tuple | the **constraint class** (`FreeQ`, ...) |
| the same predicate in ordinary Python code | the **`eager_<Name>` function** |

## 5. Adding a new node (checklist)

1. Subclass `MathematicaExpr`; implement `_evaluate` with Mathematica's semantics.
2. If a standard-SymPy twin exists, implement `rewrite_as_standard_sympy` — and verify
   the translation **numerically** against Mathematica (see the traps in §1).
3. If rules need the predicate eagerly, add `eager_<Name>` and keep the bare name on
   the class.
4. Register the head for pattern matching through `sympy_matching`'s registry
   ([`conversion.md`](../../sympy_matching/docs/conversion.md) §4) — never ad hoc.
5. Pin the behaviour with tests whose expected values come from Mathematica, not from
   the port itself.

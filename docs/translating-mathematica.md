# Translating Mathematica: from source text to SymPy

The full pipeline: Mathematica notation → Full-Form List (`mathematica_to_ffl`) →
Python/SymPy code (`ffl_to_sympy_code`) → live SymPy objects. For what the resulting
pattern objects *mean* when matching, see
[`../../sympy_matching/docs/wildcards.md`](../../sympy_matching/docs/wildcards.md) —
this package builds on `sympy_matching` (never the other way round).

Every example is a doctest, executed by `sympy_wolfram/tests/test_docs.py`.

```python
>>> from sympy import Symbol
>>> from sympy_wolfram.parser import mathematica_to_ffl
>>> from sympy_wolfram.interpreter import ffl_to_sympy_code
>>> x = Symbol('x')

```

---

## 1. Stage one: text → Full-Form List

A Full-Form List (FFL) is Mathematica's `FullForm` as nested Python lists — heads as
strings, arguments as sublists or atom strings. It is the lingua franca of this
package: everything downstream consumes FFL, so translations can also start from
pre-extracted FFL (e.g. a JSON dump produced by Mathematica itself) without any
text parsing.

```python
>>> mathematica_to_ffl("a*x^2")
['Times', 'a', ['Power', 'x', '2']]
>>> mathematica_to_ffl("Sin[x]^2 + Cos[x]^2")
['Plus', ['Power', ['Sin', 'x'], '2'], ['Power', ['Cos', 'x'], '2']]

```

Pattern syntax parses too — note how `x_` and `m_.` become `Pattern`/`Optional` nodes,
the structures whose translation rules are the subject of stage two:

```python
>>> mathematica_to_ffl("x_^m_.")
['Power', ['Pattern', 'x', ['Blank']], ['Optional', ['Pattern', 'm', ['Blank']]]]

```

## 2. Stage two: FFL → SymPy code

`ffl_to_sympy_code(ffl, reserved_symbols, namespace, ...)` returns a Python source
string, the wildcard definitions it requires, and the plain symbols it saw. The code
is meant to be `eval`'d in the namespace the converter fills in:

```python
>>> ns = {}
>>> code, wild_defs, symbols = ffl_to_sympy_code(
...     mathematica_to_ffl("Sin[x]^2 + Cos[x]^2"), {'x': 'x'}, ns)
>>> code
'((sympy.sin(x))**(Integer(2)) + (sympy.cos(x))**(Integer(2)))'
>>> eval(code, ns)
sin(x)**2 + cos(x)**2

```

Two properties of the emitted code worth knowing:

* **Fully-qualified names** (`sympy.sin`, `sympy.pi`) — the code is unambiguous
  regardless of what the surrounding module imports.
* **Exact numbers** — integers become `Integer(...)`, so `Integer(1)/Integer(4)` stays
  a `Rational`; nothing silently becomes a float.

## 3. Translating patterns: the three forms

Mathematica distinguishes three things that print as one letter, and the translation
must keep them distinct (collapsing any two changes which expressions a rule fires
on — see the README's `d`/`d_`/`d_.` discussion):

| Mathematica | FFL | emitted |
|---|---|---|
| `d` | `'d'` | `Symbol('d')` |
| `d_` | `['Pattern','d',['Blank']]` | `d_ = WildSymbol('d')` |
| `d_.` | `['Optional',['Pattern','d',['Blank']]]` | `_d_ = WildSymbol('d', optional_value=IDENTITY_ELEMENT)` |

```python
>>> code, defs, _ = ffl_to_sympy_code(mathematica_to_ffl("d_ + d_.*W"))
>>> code
"(d_ + (_d_ * Symbol('W')))"
>>> sorted(defs)
["_d_ = WildSymbol('d', optional_value=IDENTITY_ELEMENT)", "d_ = WildSymbol('d')"]

```

The two `d` wildcards are distinct SymPy objects sharing one OmniMatch variable name —
`sympy_matching` unifies them by name at match time
([`wildcards.md`](../../sympy_matching/docs/wildcards.md) §3).

## 4. Pattern side vs replacement side

A Mathematica rule has a pattern (the `lhs` of `:=`), a replacement, and usually a
`/;` condition. Only the **pattern** contains `Pattern[...]` nodes; in the replacement
and the condition, the same variables appear as **bare atoms** and must resolve to the
bound values. The caller signals which side it is translating by pre-seeding the
wildcard names:

```python
>>> # pattern side: no seeding -- bare atoms stay literal Symbols
>>> ffl_to_sympy_code(mathematica_to_ffl("d + d_.*W"))[0]
"(Symbol('d') + (_d_ * Symbol('W')))"
>>> # replacement side: seeded -- bare atoms become wildcard references
>>> ffl_to_sympy_code(mathematica_to_ffl("d*W"), optional_wildcards={'d'})[0]
"(_d_ * Symbol('W'))"
>>> ffl_to_sympy_code(mathematica_to_ffl("d*W"), wildcards={'d'})[0]
"(d_ * Symbol('W'))"

```

## 5. `reserved_symbols`: names the caller binds

The classic case is the integration variable, bound by `x_Symbol` in every rule head.
A reserved name is emitted as its bare Python identifier — never a wildcard, never a
fresh `Symbol`:

```python
>>> ffl_to_sympy_code(mathematica_to_ffl("a*x"), {'x': 'x'}, wildcards={'a'})[0]
'(a_ * x)'

```

## 6. A worked rule translation

Putting the sides together for `Int[x_^m_., x_Symbol] := x^(m+1)/(m+1) /; NeQ[m, -1]`
(the constraint side works exactly like the replacement side — bare atoms, seeded):

```python
>>> ns = {}
>>> pat_code, defs, _ = ffl_to_sympy_code(
...     mathematica_to_ffl("x_^m_."), {'x': 'x'}, ns)
>>> pat_code
'(x)**(_m_)'
>>> rep_code, _, _ = ffl_to_sympy_code(
...     mathematica_to_ffl("x^(m+1)/(m+1)"), {'x': 'x'}, ns,
...     optional_wildcards={'m'})
>>> pattern, replacement = eval(pat_code, ns), eval(rep_code, ns)

```

These SymPy objects drop straight into the `sympy_matching` rule machinery
([`rules-and-constraints.md`](../../sympy_matching/docs/rules-and-constraints.md)):

```python
>>> from sympy_matching.matching_rule import (
...     SymPyReplacementPattern, build_replacer, to_omnimatch_expression)
>>> from sympy_matching.conversion import omnimatch_to_sympy
>>> from sympy import Ne
>>> rule = SymPyReplacementPattern(
...     pattern=pattern, constraints=(Ne(ns['_m_'], -1),), replacement=replacement,
...     module_name='docs example', rule_number=1)
>>> rep = build_replacer([rule])
>>> omnimatch_to_sympy(rep.replace(to_omnimatch_expression(x**3))[0])
x**4/4

```

## 7. Layering

`sympy_wolfram` imports from `sympy_matching` (wildcards, conversion, rule machinery)
and adds the Wolfram-specific knowledge: parsing, the FFL→SymPy translation rules, the
function-name tables, and Wolfram-semantics nodes
([`nodes-and-evaluation.md`](nodes-and-evaluation.md)). Nothing in `sympy_matching`
knows this package exists — it sees only ordinary SymPy expressions, `WildSymbol`s,
and heads registered through its public registry.

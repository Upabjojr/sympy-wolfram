# `sympy_wolfram` — translating Wolfram `FullForm` into SymPy

> **⚠️ Experimental** — this package is under active development; APIs, rule
> content and behaviour may change without notice. Version 0.0.2 is a
> pre-alpha snapshot.

This package turns Mathematica expressions into SymPy ones. Its most delicate job is
translating **patterns**, because Mathematica distinguishes three things that all print
as the same letter, and collapsing any two of them silently changes which expressions a
rule fires on.

Every example below is a doctest, executed by `sympy_wolfram/tests/test_readme.py`.

For what the resulting objects then *mean* when matching, see
`sympy_matching/README.md`.

---

## 1. The entry point

`ffl_to_sympy_code` takes a Full-Form List (nested Python lists mirroring Mathematica's
`FullForm`) and returns Python source, the wildcard definitions it needs, and the plain
symbols it saw.

```python
>>> from sympy_wolfram.interpreter import ffl_to_sympy_code
>>> code, wild_defs, symbols = ffl_to_sympy_code(['Power', 'x', '2'], {'x': 'x'})
>>> code
'(x)**(Integer(2))'

```

The returned code is meant to be `eval`'d in a namespace the converter fills in for you:

```python
>>> ns = {}
>>> code, _, _ = ffl_to_sympy_code(['Plus', 'x', '1'], {'x': 'x'}, ns)
>>> eval(code, ns)
x + 1

```

---

## 2. The three pattern forms

| Mathematica | `FullForm` | translation |
|---|---|---|
| `d` | `d` | `Symbol('d')` — a literal |
| `d_` | `Pattern[d, Blank[]]` | `d_` — `WildSymbol('d')` |
| `d_.` | `Optional[Pattern[d, Blank[]]]` | `_d_` — `WildSymbol('d', optional_value=IDENTITY_ELEMENT)` |

Note the naming convention: the **plain** Blank becomes `d_` (trailing underscore) and
the **Optional** Blank becomes `_d_` (leading *and* trailing). Both carry the OmniMatch
variable name `d`, which is what unifies them — see `sympy_matching/README.md` §2.

```python
>>> blank = ['Pattern', 'd', ['Blank']]
>>> optional = ['Optional', ['Pattern', 'd', ['Blank']]]
>>> code, defs, _ = ffl_to_sympy_code(['Plus', blank, ['Times', optional, 'W']])
>>> code
"(d_ + (_d_ * Symbol('W')))"
>>> sorted(defs)
["_d_ = WildSymbol('d', optional_value=IDENTITY_ELEMENT)", "d_ = WildSymbol('d')"]

```

This is Rubi's `d_ + d_.*ProductLog[...]` shape: a denominator `d*(1 + W)` where both
occurrences are the same variable.

---

## 3. Inside a pattern, a bare atom is a LITERAL

Mathematica keeps a literal symbol independent of a same-named pattern variable —
`MatchQ[d + 5 W, d + d_.*W]` is `True`, because the literal `d` matches itself while the
variable binds 5. So a bare atom appearing in a pattern must **not** be rewritten into
the wildcard:

```python
>>> code, _, _ = ffl_to_sympy_code(['Plus', 'd', ['Times', optional, 'W']])
>>> code
"(Symbol('d') + (_d_ * Symbol('W')))"

```

`Plus` is orderless, so the very same Mathematica expression can arrive with its
arguments the other way round. The translation must be identical:

```python
>>> code, _, _ = ffl_to_sympy_code(['Plus', ['Times', optional, 'W'], 'd'])
>>> code
"((_d_ * Symbol('W')) + Symbol('d'))"

```

Both keep `Symbol('d')` a literal. (This used to be order-dependent: the sets of
discovered wildcards fill up as the converter walks the tree, so a literal appearing
*after* the wildcard was rewritten *into* it, producing a pattern that demanded both be
equal and no longer matched `d + 5 W`.)

---

## 4. Pattern side vs replacement side

A rule's **replacement** and **constraints** contain no `Pattern[...]` nodes at all —
their wildcards appear as bare atoms, and there they *must* resolve to the bound
values. The caller signals this by pre-seeding the names:

```python
>>> code, _, _ = ffl_to_sympy_code(['Times', 'd', 'W'], wildcards={'d'})
>>> code
"(d_ * Symbol('W'))"
>>> code, _, _ = ffl_to_sympy_code(['Times', 'd', 'W'], optional_wildcards={'d'})
>>> code
"(_d_ * Symbol('W'))"

```

So the rule is:

* **no** `wildcards=` / `optional_wildcards=` given → this is a **pattern**; bare atoms
  stay literal `Symbol`s;
* names given → this is a **replacement/constraint**; bare atoms resolve to wildcards.

---

## 5. `reserved_symbols` — names bound by the caller

A name the caller already binds (classically the integration variable, from
`x_Symbol`) must never become a wildcard. Pass it in `reserved_symbols`:

```python
>>> code, _, _ = ffl_to_sympy_code(['Times', 'a', 'x'], {'x': 'x'}, wildcards={'a'})
>>> code
'(a_ * x)'

```

`x` is emitted as the bare identifier `x`, not `Symbol('x')` and not a wildcard.

There is a subtlety worth knowing: `x_.` where `x` is *also* reserved. Both bind the
same name, so the "absent" branch would have to give `x` the `Times` identity 1, which
then fails `x_Symbol`. The optional branch is therefore unreachable and the factor is in
fact mandatory — verified in Mathematica, where `g[x_.*h[x_], x_Symbol]` matches
`g[z h[z], z]` but **not** `g[h[z], z]`.

---

## 6. Constants, numbers and heads

Known constants and functions are emitted fully qualified, so the generated code is
unambiguous no matter what the surrounding module has imported:

```python
>>> ffl_to_sympy_code('Pi')[0]
'sympy.pi'
>>> ffl_to_sympy_code(['Sin', 'x'], {'x': 'x'})[0]
'sympy.sin(x)'
>>> ffl_to_sympy_code(['Power', 'x', '2'], {'x': 'x'})[0]
'(x)**(Integer(2))'

```

Integers become `Integer(...)` rather than Python `int`s, so exact arithmetic is
preserved (`Integer(1)/Integer(4)` is a `Rational`, whereas `1/4` would be a float):

```python
>>> ns = {}
>>> code, _, _ = ffl_to_sympy_code(['Times', '3', 'x'], {'x': 'x'}, ns)
>>> code
'(Integer(3) * x)'
>>> eval(code, ns)
3*x

```

---

## 7. Round trip: translate, then match

Putting §2–§4 together — translate a Wolfram pattern and use it. The values below are
Mathematica's own answers for `MatchQ[..., d + d_.*W]`:

```python
>>> from sympy import Symbol
>>> from omnimatch import ManyToOneMatcher, Pattern
>>> from sympy_matching.matching_rule import to_omnimatch_expression
>>> x, W, d = Symbol('x'), Symbol('W'), Symbol('d')
>>> ns = {}
>>> code, _, _ = ffl_to_sympy_code(['Plus', 'd', ['Times', optional, 'W']], namespace=ns)
>>> pattern = eval(code, ns)
>>> def matches(pat, subject):
...     m = ManyToOneMatcher()
...     m.add(Pattern(to_omnimatch_expression(pat)))
...     return bool(list(m.match(to_omnimatch_expression(subject))))
>>> matches(pattern, d + d*W)      # Mathematica: True
True
>>> matches(pattern, 5 + 5*W)      # Mathematica: False -- literal d is not 5
False
>>> matches(pattern, d + 5*W)      # Mathematica: True  -- variable binds 5
True

```

---

## Why this matters

A rule that fires where Mathematica's would not produces a *wrong antiderivative*, not
an error — the replacement derived for one shape gets applied to another. The three
forms therefore have to stay distinct all the way from `FullForm` to the OmniMatch
pattern, which is why they are pinned by tests at both layers:

* `sympy_wolfram/tests/test_blank_optional_semantics.py` — 37 cases, every expected
  value read directly off Mathematica 12.2;
* `sympy_wolfram/tests/test_docs.py` and `sympy_matching/tests/test_docs.py` — these
  documents.

## See also

* [`docs/translating-mathematica.md`](docs/translating-mathematica.md) — the full
  pipeline: notation → Full-Form List → SymPy code → live rule, with a worked example.
* [`docs/nodes-and-evaluation.md`](docs/nodes-and-evaluation.md) — deferred
  `MathematicaExpr` nodes, `doit()` vs `rewrite_as_standard_sympy()`, `eager_<Name>`
  functions, and the constraint classes.
* `sympy_matching/docs/` — the matching layer this package builds on.

## Testing

`pytest tests/` — a handful of tests exercise the re-exports that the
higher-level `rubi-integrate` package builds on top of this one; they need
`rubi-integrate` importable (it is not a runtime dependency of this package).

## License

MIT License, Copyright (c) 2026 Francesco Bonazzi. See `LICENSE`.

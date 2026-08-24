# -*- coding: utf-8 -*-
"""Interpreter: Full-Form List (FFL) -> SymPy code and objects.

This is where MEANING is assigned. The parser produces a purely syntactic FFL;
this module decides what each Wolfram head becomes in SymPy, which names are
pattern wildcards, and what the evaluation namespace contains.

    parser.py       text     -> FFL
    interpreter.py  FFL      -> SymPy    (this module)
    objects.py      the Mathematica objects this interpreter can emit

No dependency on rubi_integrate or any other domain-specific package.

Wildcard naming convention:
    - Pattern[name, Blank[]]        -> ``name_``   (non-optional wildcard)
    - Optional[Pattern[name, ...]]  -> ``_name_``  (optional wildcard, identity element)

The generated code strings can be ``eval``'d in a namespace containing SymPy
objects and the appropriate wildcard symbols.
"""
from __future__ import annotations

import keyword
import warnings
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

import sympy
from sympy import Integer, Rational, Symbol
from sympy.core.function import AppliedUndef
from sympy.printing.str import StrPrinter

from sympy_matching.wild import (HeadRef, IDENTITY_ELEMENT, WildHeadApp, WildHeadDeriv,
                                 WildSymbol)

from .parser import mathematica_to_ffl

# Type alias for the custom_functions dict expected by the public APIs.
# Maps Wolfram head name -> (qualified_code_str, python_object)
CustomFunctionsDict = Dict[str, Tuple[str, Any]]


# =============================================================================
# Wild printer (stateless module-level instance)
# =============================================================================

class _WildStrPrinter(StrPrinter):
    """Custom StrPrinter that prints WildSymbol using variable-name convention.

    Non-optional WildSymbol('m') -> ``m_``
    Optional WildSymbol('m', optional_value=IDENTITY_ELEMENT) -> ``_m_``
    """

    # Names of undefined functions (``sympy.Function('Head')`` placeholders) that the
    # evaluating namespace does NOT define, so a bare ``Head(...)`` could not be
    # re-evaluated: print those qualified. Set per call by :func:`_simplify_code`;
    # the module-level instance keeps it empty.
    _qualify_undefined: FrozenSet[str] = frozenset()
    # Names of plain Symbols that the evaluating namespace binds to SOMETHING ELSE
    # (a coefficient named ``D`` vs the Wolfram ``D`` node): print ``Symbol('D')``.
    _qualify_symbols: FrozenSet[str] = frozenset()

    def _print_WildSymbol(self, expr):
        name = expr.wildcard_name
        if expr.optional_value is not None:
            return f'_{name}_'
        return f'{name}_'

    def _print_Symbol(self, expr):
        if expr.name in self._qualify_symbols:
            return f"Symbol('{expr.name}')"
        return super()._print_Symbol(expr)

    def _print_Exp1(self, expr):
        # Bare ``E`` is deliberately NOT in the shortening namespace (it could be a
        # coefficient symbol), so print the constant qualified.
        return 'sympy.E'

    def _print_AppliedUndef(self, expr):
        name = type(expr).__name__
        if name in self._qualify_undefined:
            args = ', '.join(self._print(a) for a in expr.args)
            return f"sympy.Function('{name}')({args})"
        return super()._print_Function(expr)

    def _print_Half(self, expr):
        return "sympy.S.Half"

    def _print_Rational(self, expr: Rational):
        num = self._print(expr.numerator)
        denom = self._print(expr.denominator)
        return f"sympy.S({num})/{denom}"


# Module-level printer instance (stateless)
_wild_printer = _WildStrPrinter()


def _is_numeric_atom(atom: str) -> bool:
    """True for an FFL atom that is a number literal (``'2'``, ``'-1'``, ``'0.5'``)."""
    try:
        float(atom)
        return True
    except ValueError:
        return False


# =============================================================================
# FFLConverter
# =============================================================================

class FFLConverter:
    """Convert Mathematica FFL AST nodes to SymPy code strings.

    Parameters
    ----------
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names that are BOUND
        externally by the caller and must therefore never be turned into pattern
        wildcards. ``Pattern[name, ...]`` and ``Optional[Pattern[name, ...]]``
        for such a name both collapse to the plain identifier.
    custom_functions : dict, optional
        Mapping from Wolfram/Mathematica head names to custom callables.
        Each value is a 2-tuple ``(qualified_code_str, obj)`` where:

        * *qualified_code_str* -- the Python code string to emit
          (e.g. ``"my_module.MyFunc"``).
        * *obj* -- the actual Python object for the eval namespace.  If the
          code string contains a dot (e.g. ``"mod.Func"``), *obj* should be
          the module (added as ``mod``).  If there is no dot, *obj* should be
          the callable itself.
    extra_sympy_funcs : dict, optional
        Additional ``{MathematicaHead: 'python.callable'}`` mappings.
    extra_constants : dict, optional
        Additional ``{MathematicaAtom: 'sympy.constant_code'}`` mappings.
    """

    # Type alias for custom_functions dict value
    CustomFuncEntry = Tuple[str, Any]

    # Mathematica head -> SymPy callable code
    SYMPY_FUNC_MAP: Dict[str, str] = {
        # Wildcard function head applied to args (F_[v_]) -- see WildHeadApp
        'WildHeadApp': 'WildHeadApp',
        'WildHeadDeriv': 'WildHeadDeriv',
        # Trigonometric
        'Sin': 'sympy.sin', 'Cos': 'sympy.cos', 'Tan': 'sympy.tan',
        'Sec': 'sympy.sec', 'Csc': 'sympy.csc', 'Cot': 'sympy.cot',
        'ArcSin': 'sympy.asin', 'ArcCos': 'sympy.acos', 'ArcTan': 'sympy.atan',
        'ArcSec': 'sympy.asec', 'ArcCsc': 'sympy.acsc', 'ArcCot': 'sympy.acot',
        # Lowercase trig (some FFL use them)
        'sin': 'sympy.sin', 'cos': 'sympy.cos', 'tan': 'sympy.tan',
        'sec': 'sympy.sec', 'csc': 'sympy.csc', 'cot': 'sympy.cot',
        'sinh': 'sympy.sinh', 'cosh': 'sympy.cosh', 'tanh': 'sympy.tanh',
        'sech': 'sympy.sech', 'csch': 'sympy.csch', 'coth': 'sympy.coth',
        # Hyperbolic
        'Sinh': 'sympy.sinh', 'Cosh': 'sympy.cosh', 'Tanh': 'sympy.tanh',
        'Sech': 'sympy.sech', 'Csch': 'sympy.csch', 'Coth': 'sympy.coth',
        'ArcSinh': 'sympy.asinh', 'ArcCosh': 'sympy.acosh', 'ArcTanh': 'sympy.atanh',
        'ArcSech': 'sympy.asech', 'ArcCsch': 'sympy.acsch', 'ArcCoth': 'sympy.acoth',
        # Elementary
        'Exp': 'sympy.exp', 'Log': 'sympy.log', 'Sqrt': 'sympy.sqrt',
        'Abs': 'sympy.Abs',
        # Special functions
        'EllipticE': 'sympy.elliptic_e', 'EllipticF': 'sympy.elliptic_f',
        'EllipticK': 'sympy.elliptic_k',
        # EllipticPi has a node (arity-overloaded: [n, m] / [n, phi, m]).
        'AppellF1': 'sympy.appellf1',
        # HypergeometricPFQ[{a1,..}, {b1,..}, z] -> hyper([a1,..], [b1,..], z): the
        # List arguments are emitted as Python list literals, which hyper accepts.
        # Without this entry the generic sympy.Function('HypergeometricPFQ') fallback
        # was emitted, and it CRASHES at eval time on the list arguments -- every
        # generated module containing pFq failed to import.
        'HypergeometricPFQ': 'sympy.hyper',
        'Gamma': 'Gamma', 'LogGamma': 'sympy.loggamma',
        'Erf': 'sympy.erf',
        'Erfi': 'sympy.erfi', 'Erfc': 'sympy.erfc',
        'FresnelS': 'sympy.fresnels', 'FresnelC': 'sympy.fresnelc',
        'SinIntegral': 'sympy.Si', 'CosIntegral': 'sympy.Ci',
        # NOTE: BesselJ / ExpIntegralE / PolyGamma / Zeta / Factorial are deliberately
        # NOT renamed to sympy.besselj/expint/polygamma/zeta/factorial here. They have
        # nodes in this package, and a Wolfram head that we implement keeps its own
        # node -- SymPy's versions apply their own eager-evaluation rules, which are
        # not Mathematica's. They are picked up by wolfram_library_names().
        'SinhIntegral': 'sympy.Shi', 'CoshIntegral': 'sympy.Chi',
        'PolyLog': 'sympy.polylog',
        # Calculus / algebra
        'D': 'D', 'Denominator': 'sympy.denom',
        # NOTE: 'Rt' is intentionally NOT here. It is a Rubi utility (see
        # rubi_integrate.codegen RUBI_UTILS_MAP), emitted as the deferred Rt(...) node so
        # its simplest-root algorithm runs at fire time -- not eagerly as sympy.root.
        'Simplify': 'Simplify',
        'FractionalPart': 'sympy.frac', 'IntegerPart': 'sympy.floor',
        # List functions:
        'Min': 'sympy.Min', 'Max': 'sympy.Max',
    }

    # Mathematica predicates that map to native SymPy relational/logic objects
    SYMPY_LOGIC_MAP: Dict[str, str] = {
        'Equal': 'sympy.Eq',
        'Unequal': 'sympy.Ne',
        'Less': 'sympy.Lt',
        'Greater': 'sympy.Gt',
        'LessEqual': 'sympy.Le',
        'GreaterEqual': 'sympy.Ge',
    }

    CONSTANT_MAP: Dict[str, str] = {
        'Pi': 'sympy.pi', 'E': 'sympy.E',
        'I': 'sympy.I', 'Infinity': 'sympy.oo',
        'True': 'sympy.true', 'False': 'sympy.false',
        'EulerGamma': 'sympy.EulerGamma',
    }

    # SymPy names that the shortening printer may emit UNQUALIFIED (e.g. ``sqrt``,
    # ``fresnels``, ``Eq``) rather than as ``sympy.sqrt``. A few structural helpers the
    # emitter always relies on but that are not reachable through the translation maps.
    _GENERATED_SYMPY_EXTRAS: Tuple[str, ...] = (
        'sqrt', 'exp', 'log', 'Abs', 'pi', 'I', 'oo',
        'root', 'diff', 'simplify', 'hyper', 'atan2',
        # Every SymPy function `MathematicaExpr.rewrite_as_standard_sympy()` can
        # produce. A caller may bake that translation into generated code (see the
        # `rewrite` hook of ffl_to_sympy_short_code), and the printed result names
        # these BARE -- so they must resolve both in the generated module and in the
        # shortening namespace, or the round-trip fails and the rewrite is silently
        # discarded.
        'Ei', 'li', 'LambertW',
        'gamma', 'uppergamma', 'factorial', 'zeta', 'polygamma', 'besselj', 'expint',
        'elliptic_pi', 'Derivative', 'Subs',
        # The printer emits this constant BARE (unlike ``sympy.E``, it cannot collide
        # with a coefficient symbol); without it every expression containing
        # EulerGamma failed the round-trip and stayed verbose.
        'EulerGamma',
    )

    # Map targets that must NOT be exposed as bare names: the generated header binds
    # ``Min``/``Max`` to ``Symbol('Min')``/``Symbol('Max')`` (Rubi passes them as bare
    # ordering flags, e.g. ``Expon[Px, x, Min]``; a genuine ``Min[a, b]`` call is emitted
    # qualified as ``sympy.Min(...)``). So bare ``Min`` is a Symbol in the file, never the
    # SymPy function -- keep it out of the shortening namespace to match.
    _GENERATED_SYMPY_EXCLUDE: frozenset = frozenset({'Min', 'Max'})

    @classmethod
    def wolfram_library_names(cls) -> Dict[str, Any]:
        """Every Wolfram node this package provides, as ``{Wolfram name: class}``.

        ``sympy_wolfram`` is an interpreter for the Wolfram language AND the runtime
        library that translated code links against, so a head with a node here is
        emitted as THAT node -- not silently renamed to a SymPy function that merely
        looks similar. Keeping them separate matters because SymPy's functions apply
        their own eager-evaluation rules, which are not Mathematica's.

        Discovered by walking the package rather than listed by hand: adding a node
        makes it translatable and round-trippable with no second place to update.
        """
        cached = getattr(cls, '_WOLFRAM_LIBRARY_CACHE', None)
        if cached is not None:
            return cached
        from sympy_wolfram import mathematica_functions as _mf
        from sympy_wolfram import objects as _ob
        registry: Dict[str, Any] = {}
        for module in (_ob, _mf):
            for name, obj in vars(module).items():
                if name.startswith('_') or not isinstance(obj, type):
                    continue
                if issubclass(obj, _ob.MathematicaExpr) and obj is not _ob.MathematicaExpr:
                    registry[name] = obj
        cls._WOLFRAM_LIBRARY_CACHE = registry
        return registry

    @classmethod
    def generated_code_sympy_names(cls) -> Dict[str, Any]:
        """SINGLE SOURCE OF TRUTH for the bare SymPy names available in generated rule
        code -- and therefore in the shortening eval namespace.

        Both :attr:`_eval_ns` (used to verify a shortened form re-evaluates equal) AND
        the generated-file ``from sympy import (...)`` header (see
        ``rubi_integrate.codegen.generate._sympy_import_line``) are built from this dict, so
        the two can never drift: a function reachable through the translation maps is
        importable in the generated module IFF it is evaluable during shortening. (If
        the shortening namespace had a name the header lacked, the shortener would emit a
        bare call the per-rule load probe -- which uses the header -- then rejects,
        silently skipping the rule; the reverse just leaves rules needlessly verbose.)

        Rubi utilities and constraint predicates are deliberately NOT here: they are
        registered per call as unevaluated ``Function`` placeholders so shortening keeps
        their call-form (``FreeQ([a, b], x)``, ``And(...)``) instead of letting the real
        classes rewrite them (list->tuple, ``And``->``&``).
        """
        names: Dict[str, Any] = {}
        # Functions and relational heads reachable through the translation maps. The
        # CONSTANT_MAP is deliberately NOT included: pi/I/oo/EulerGamma are covered by
        # the extras below, while E stays qualified (``sympy.E``) -- a bare ``E`` would
        # both be dead weight and risk colliding with a coefficient symbol named E.
        for target in list(cls.SYMPY_FUNC_MAP.values()) + list(cls.SYMPY_LOGIC_MAP.values()):
            if not target.startswith('sympy.'):
                continue
            bare = target.split('.', 1)[1]
            if bare in cls._GENERATED_SYMPY_EXCLUDE:
                continue
            obj = getattr(sympy, bare, None)
            if obj is not None:
                names[bare] = obj
        for bare in cls._GENERATED_SYMPY_EXTRAS:
            obj = getattr(sympy, bare, None)
            if obj is not None:
                names[bare] = obj
        return names

    # Function heads that may appear as a BARE atom — i.e. as a *value* rather than an
    # application — in a head test such as MemberQ[{ArcSin, ArcCos, ...}, F] or
    # EqQ[F, Sin], where F is a function-head wildcard bound to a real function. Emitting
    # these as the SymPy class (``sympy.asin``) rather than ``Symbol('ArcSin')`` lets the
    # test compare against the bound head (a HeadRef carrying that class) instead of an
    # unrelated symbol. Restricted to genuine function heads: NOT Min/Max (used as an
    # ordering sentinel in Exponent[u, x, Min]) or structural heads (D/Simplify/And/...).
    _HEAD_FUNCTION_NAMES: frozenset = frozenset({
        'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch',
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
        'ArcSin', 'ArcCos', 'ArcTan', 'ArcCot', 'ArcSec', 'ArcCsc',
        'ArcSinh', 'ArcCosh', 'ArcTanh', 'ArcCoth', 'ArcSech', 'ArcCsch',
        'Erf', 'Erfc', 'Erfi', 'FresnelS', 'FresnelC',
        'ExpIntegralEi', 'LogIntegral',
        'SinIntegral', 'CosIntegral', 'SinhIntegral', 'CoshIntegral',
    })

    def __init__(
        self,
        reserved_symbols: Optional[Mapping[str, str]] = None,
        custom_functions: Optional[Dict[str, "FFLConverter.CustomFuncEntry"]] = None,
        extra_sympy_funcs: Optional[Dict[str, str]] = None,
        extra_constants: Optional[Dict[str, str]] = None,
    ) -> None:
        self._reserved_symbols: Dict[str, str] = dict(reserved_symbols or {})
        # Per-rule wildcard tracking (reset per rule via reset())
        self._wildcards_non_optional: Set[str] = set()
        self._wildcards_optional: Set[str] = set()
        self._symbols: Set[str] = set()
        # Names declared as locals of an enclosing Module/With/Block binding list.
        # These (and only these) are emitted bare and declared at the module top.
        self._scope_locals: Set[str] = set()
        self._bare_locals: Set[str] = set()
        self._wild_defs: List[str] = []
        # Slot mapping for Function/Lambda conversion (slot_number -> var_name)
        self._slot_vars: Dict[str, str] = {}
        self._custom_functions: Dict[str, Tuple[str, Any]] = custom_functions or {}

        self.func_map: Dict[str, str] = {**self.SYMPY_FUNC_MAP, **self.SYMPY_LOGIC_MAP}
        # Allow caller to extend the maps
        if extra_sympy_funcs:
            self.func_map = {**self.func_map, **extra_sympy_funcs}
        if extra_constants:
            self.CONSTANT_MAP = {**self.CONSTANT_MAP, **extra_constants}

        # Namespace the emitted code is evaluated against (see `namespace`).
        # Built here so every converter starts from the same base names; the
        # caller's own entries are layered on top by `use_namespace`.
        self._eval_ns: Dict[str, Any] = {
            'sympy': sympy, 'Integer': Integer, 'Rational': Rational,
            'Symbol': Symbol, 'WildSymbol': WildSymbol,
            'WildHeadApp': WildHeadApp, 'WildHeadDeriv': WildHeadDeriv,
            'HeadRef': HeadRef,
            'IDENTITY_ELEMENT': IDENTITY_ELEMENT,
            'x': Symbol('x'),
            # Every SymPy function/constant the generated code may reference unqualified,
            # from the SINGLE SOURCE shared with the generated-file import header, so the
            # two never drift (see generated_code_sympy_names).
            **self.generated_code_sympy_names(),
            # The Wolfram runtime library, so the shortening round-trip can eval a
            # node the emitter just wrote. Without this, any rule mentioning e.g.
            # ProductLog failed to eval and silently kept its VERBOSE form.
            **self.wolfram_library_names(),
            # Logical operators — use unevaluated wrappers so simplify_code
            # round-trip preserves And(...)/Or(...)/Not(...) form rather than
            # the &/|/~ infix operators that sympy.And/Or/Not would produce.
            'And': sympy.Function('And'),
            'Or': sympy.Function('Or'),
            'Not': sympy.Function('Not'),
        }

        # Register custom functions in the eval namespace
        for wolfram_name, (code_str, obj) in self._custom_functions.items():
            if obj is None:
                continue
            if '.' in code_str:
                # Only register the module prefix (e.g. 'my_module' for a code_str
                # of 'my_module.Foo') so that eval('my_module.Foo(...)') works.
                # Do NOT register the short
                # name ('Subst') — the generated code uses the qualified form and
                # simplify_code must preserve it.
                ns_key = code_str.split('.')[0]
                self._eval_ns[ns_key] = obj
            else:
                self._eval_ns[code_str] = obj

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def reserved_symbols(self) -> Dict[str, str]:
        """Wolfram symbol name -> Python identifier, for names that are BOUND
        externally and must therefore never become pattern wildcards."""
        return self._reserved_symbols

    @reserved_symbols.setter
    def reserved_symbols(self, val: Optional[Mapping[str, str]]) -> None:
        self._reserved_symbols = dict(val or {})

    @property
    def wildcards_non_optional(self) -> Set[str]:
        return self._wildcards_non_optional

    @property
    def wildcards_optional(self) -> Set[str]:
        return self._wildcards_optional

    @property
    def eval_ns(self) -> Dict[str, Any]:
        return self._eval_ns

    def use_namespace(self, namespace: Dict[str, Any]) -> Dict[str, Any]:
        """Adopt *namespace* as this converter's evaluation namespace, in place.

        The dict is seeded with the converter's base names, then the caller's own
        entries are restored on top (so a caller can override any base name), and
        finally every wildcard discovered during conversion is written into it.
        The SAME dict object is returned, so after converting it can `eval` the
        emitted code.
        """
        caller_entries = dict(namespace)
        namespace.update(self._eval_ns)
        namespace.update(caller_entries)
        self._eval_ns = namespace
        return namespace

    @property
    def wild_defs(self) -> List[str]:
        """Variable definition strings for WildSymbol declarations."""
        return list(self._wild_defs)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-rule state (wildcard sets and defs)."""
        self._wildcards_non_optional = set()
        self._wildcards_optional = set()
        self._symbols = set()
        self._scope_locals = set()
        self._bare_locals = set()
        self._wild_defs = []

    def convert(self, ffl: Any, *, is_pattern: bool = False) -> str:
        """Convert an FFL node to a SymPy code string.

        Parameters
        ----------
        ffl : list or str
            A Mathematica Full-Form List node.
        is_pattern : bool
            If True, Pattern/Optional nodes produce wildcard references.

        Returns
        -------
        str
            Python code string that evaluates to a SymPy expression.
        """
        if isinstance(ffl, str):
            return self._atom_to_code(ffl, is_pattern)

        if not isinstance(ffl, list) or not ffl:
            return repr(ffl)

        head = ffl[0]

        # -- Derivative[n, ...][f][args] (f'[x], f''[x]) --------------------------
        # A nested head: the parser gives [[['Derivative', '1'], 'f'], 'x']. With
        # non-negative integer orders this is sympy.Derivative(f(args), (arg, n), ...).
        # Symbolic or negative orders (Derivative[m][f], Mathematica's antiderivative
        # Derivative[-1][f]) have no SymPy counterpart and fall through to the error.
        deriv = self._derivative_to_code(head, ffl[1:], is_pattern)
        if deriv is not None:
            return deriv

        # Non-string head (e.g., F_[x_]) -- unsupported
        if not isinstance(head, str):
            raise ValueError(
                f"Non-string function head {head!r} -- function-head wildcard "
                f"patterns (e.g., F_[x_]) are not yet supported."
            )

        # -- Pattern / Optional ------------------------------------------------
        if head == 'Pattern':
            return self._pattern_to_code(ffl)
        if head == 'Optional':
            return self._optional_to_code(ffl)

        # -- Slot (pure function argument placeholder) --------------------------
        if head == 'Slot':
            slot_num = ffl[1] if len(ffl) > 1 else '1'
            if slot_num in self._slot_vars:
                return self._slot_vars[slot_num]
            # Outside a Function context, fall back to a generic representation
            return f"Symbol('xi{slot_num}')"

        # -- Function (pure function / Lambda) ---------------------------------
        # Function[body]         -> Lambda(xi1, body_with_Slot1_replaced)
        # Function[param, body]  -> Lambda(param, body)
        # Function[{p1,p2}, body] -> Lambda((p1,p2), body)
        if head == 'Function':
            return self._function_to_code(ffl, is_pattern=is_pattern)

        # -- Arithmetic --------------------------------------------------------
        if head == 'Plus':
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"({' + '.join(args)})"
        if head == 'Times':
            if len(ffl) > 1 and ffl[1] == '1':
                if len(ffl) == 3:
                    return self.convert(ffl[2], is_pattern=is_pattern)
                args = [self.convert(a, is_pattern=is_pattern) for a in ffl[2:]]
            else:
                args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"({' * '.join(args)})"
        if head == 'Power':
            base = self.convert(ffl[1], is_pattern=is_pattern)
            exp_code = self.convert(ffl[2], is_pattern=is_pattern)
            return f"({base})**({exp_code})"

        # -- Boolean / logical operators ---------------------------------------
        # Emit as bare And(...)/Or(...)/Not(...) calls so the generated code is
        # independent of the execution namespace (the caller decides whether
        # And means sympy.And, a constraint combiner, etc.).
        # The eval_ns maps these to sympy.Function wrappers so that
        # simplify_code round-trips correctly without &/|/~ infix rewriting.
        if head == 'And':
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"And({', '.join(args)})"
        if head == 'Or':
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"Or({', '.join(args)})"
        if head == 'Not':
            arg = self.convert(ffl[1], is_pattern=is_pattern)
            return f"Not({arg})"

        # -- ArcTan (1 or 2 args) ----------------------------------------------
        if head == 'ArcTan':
            if len(ffl) == 3:
                x_arg = self.convert(ffl[1], is_pattern=is_pattern)
                y_arg = self.convert(ffl[2], is_pattern=is_pattern)
                return f'sympy.atan2({y_arg}, {x_arg})'
            arg = self.convert(ffl[1], is_pattern=is_pattern)
            return f'sympy.atan({arg})'

        # -- Log (1 or 2 args) -------------------------------------------------
        # Mathematica Log[b, z] is the logarithm of z to base b; SymPy's log(z, b)
        # takes the same two arguments in the OPPOSITE order. A bare name mapping
        # would silently emit log(b, z) == log(b)/log(z), the reciprocal.
        if head == 'Log' and len(ffl) == 3 and 'Log' not in self._custom_functions:
            base = self.convert(ffl[1], is_pattern=is_pattern)
            z = self.convert(ffl[2], is_pattern=is_pattern)
            return f'sympy.log({z}, {base})'

        # -- Hypergeometric2F1 -------------------------------------------------
        if head == 'Hypergeometric2F1' and len(ffl) == 5:
            a = self.convert(ffl[1], is_pattern=is_pattern)
            b = self.convert(ffl[2], is_pattern=is_pattern)
            c = self.convert(ffl[3], is_pattern=is_pattern)
            z = self.convert(ffl[4], is_pattern=is_pattern)
            return f"sympy.hyper([{a}, {b}], [{c}], {z})"

        # -- List: custom_functions override or Python list literal -------------
        if head == 'List':
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            if head in self._custom_functions:
                code_str, _ = self._custom_functions[head]
                return f"{code_str}({', '.join(args)})"
            return f"[{', '.join(args)}]"

        # -- N-ary comparison operators (chain with And) -----------------------
        # MUST be checked before custom_functions: LessEqual[3, Denom[p], 4]
        # has 3 args and must become And(LeQ(3,x), LeQ(x,4)), not LeQ(3,x,4).
        # In constraint context the caller registers LessEqual→LeQ etc. via
        # custom_functions; we honour that mapping so the chain uses the deferred
        # constraint class instead of an eagerly-evaluated sympy.Le.
        _CHAINED_COMPARISON_WOLFRAM = {'Less', 'Greater', 'LessEqual', 'GreaterEqual'}
        _CHAINED_COMPARISON_DEFAULT = {
            'Less': 'sympy.Lt',
            'Greater': 'sympy.Gt',
            'LessEqual': 'sympy.Le',
            'GreaterEqual': 'sympy.Ge',
        }
        if head in _CHAINED_COMPARISON_WOLFRAM and len(ffl) > 3:
            # Use custom mapping if provided (e.g. LeQ in constraint context),
            # otherwise fall back to the eager sympy.Le etc.
            if head in self._custom_functions:
                op = self._custom_functions[head][0]
            else:
                op = _CHAINED_COMPARISON_DEFAULT[head]
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            pairs = [f"{op}({args[i]}, {args[i+1]})" for i in range(len(args) - 1)]
            if len(pairs) == 1:
                return pairs[0]
            return f"And({', '.join(pairs)})"

        # -- Scoping constructs: register the binding-list locals (so they print BARE,
        #    declared once at the module top) and emit the binding list as a Python
        #    DICT -- `Module({r: v1, s: v2, k: None}, body)` -- instead of the verbose
        #    `List(Set(r, v1), Set(s, v2), k)`. Falls back to the generic form for a
        #    binding that is not a plain List of Set / bare-symbol items.
        if head in ('Module', 'With', 'Block') and len(ffl) >= 2:
            self._register_scope_locals(ffl[1])
            scoped = self._scoping_to_code(head, ffl, is_pattern)
            if scoped is not None:
                return scoped

        # -- Custom functions (before built-in map) ----------------------------
        if head in self._custom_functions:
            code_str, _ = self._custom_functions[head]
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"{code_str}({', '.join(args)})"

        # -- Heads that are NOT a call in SymPy --------------------------------
        # These three cannot be expressed as a name substitution, so they are built
        # structurally rather than emitted as a helper call. Emitting a call to an
        # eager helper would work, but it leaves `eager_Complex(...)` littered through
        # the generated rules; the rules should read as ordinary SymPy.
        if head == 'Complex' and len(ffl) == 3:
            # Complex[a, b] is the NUMBER a + I b, not a function.
            re_, im_ = (self.convert(a, is_pattern=is_pattern) for a in ffl[1:])
            return f"({re_} + sympy.I*{im_})"
        if head == 'Identity' and len(ffl) == 2:
            # Identity[z] == z. Rubi uses it only to stop a coefficient folding away
            # early (Int[-u_, x] := Identity[-1]*Int[u, x]).
            return self.convert(ffl[1], is_pattern=is_pattern)

        # -- Known SymPy functions ---------------------------------------------
        if head in self.func_map:
            func = self.func_map[head]
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"{func}({', '.join(args)})"

        # -- RemoveContent (pass through) --------------------------------------
        if head == 'RemoveContent':
            return self.convert(ffl[1], is_pattern=is_pattern)

        # -- A node provided by the Wolfram runtime library ---------------------
        if head in self.wolfram_library_names():
            args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
            return f"{head}({', '.join(args)})"

        # -- Generic fallback: sympy.Function('Head')(...) ---------------------
        args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
        return f"sympy.Function('{head}')({', '.join(args)})"

    # Alias for backward compatibility with code using the old class name
    ffl_to_sympy_expr = convert

    def preprocess_test_ffl(self, ffl: Any) -> Any:
        """Normalize test-suite specific constructs before conversion."""
        if not isinstance(ffl, list):
            return ffl
        if ffl[0] == 'If' and len(ffl) >= 3:
            return self.preprocess_test_ffl(ffl[2])
        return [self.preprocess_test_ffl(item) for item in ffl]

    def wildcard_ref(self, name: str) -> Optional[str]:
        """Get the Python reference for a wildcard name, or None."""
        if name in self._wildcards_non_optional:
            return f"{name}_"
        if name in self._wildcards_optional:
            return f"_{name}_"
        return None

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _scoping_to_code(self, head: str, ffl, is_pattern: bool) -> Optional[str]:
        """Emit a Module/With/Block with its binding list as a Python dict:
        ``Head({local: value, ...}, body)``. A ``None`` value denotes an
        uninitialised local. Returns None (fall back to the generic ``List(Set(...))``
        form) when the head is not mapped or a binding item is non-standard."""
        if head not in self._custom_functions:
            return None
        binding = ffl[1]
        if not (isinstance(binding, list) and binding and binding[0] == 'List'):
            return None
        entries = []
        for item in binding[1:]:
            if isinstance(item, str):
                entries.append(f"{self.convert(item, is_pattern=is_pattern)}: None")
            elif (isinstance(item, list) and len(item) == 3 and item[0] == 'Set'):
                key = self.convert(item[1], is_pattern=is_pattern)
                val = self.convert(item[2], is_pattern=is_pattern)
                entries.append(f"{key}: {val}")
            else:
                return None
        body = self.convert(ffl[2], is_pattern=is_pattern) if len(ffl) >= 3 else 'Null'
        head_code = self._custom_functions[head][0]
        return f"{head_code}({{{', '.join(entries)}}}, {body})"

    def _register_scope_locals(self, binding_ffl) -> None:
        """Record the local names of a Module/With/Block binding list so they emit
        bare. Locals are ``['Set', name, value]`` (initialised) or a bare string
        (uninitialised) inside the leading ``List``."""
        if not (isinstance(binding_ffl, list) and binding_ffl and binding_ffl[0] == 'List'):
            return
        for item in binding_ffl[1:]:
            name = None
            if isinstance(item, str):
                name = item
            elif (isinstance(item, list) and len(item) >= 2
                  and item[0] == 'Set' and isinstance(item[1], str)):
                name = item[1]
            if name and name.isidentifier() and not keyword.iskeyword(name):
                self._scope_locals.add(name)

    def _atom_to_code(self, atom: str, is_pattern: bool) -> str:
        """Convert an atom (string) to SymPy code."""
        if atom in self.CONSTANT_MAP:
            return self.CONSTANT_MAP[atom]
        try:
            n = int(atom)
            return f'Integer({n})'
        except ValueError:
            pass
        try:
            float(atom)
            return f"Rational('{atom}')"
        except ValueError:
            pass
        # Externally-bound symbol: emit its Python identifier, not a wildcard.
        if atom in self._reserved_symbols:
            return self._reserved_symbols[atom]
        # A bare function-head name used as a VALUE (e.g. in MemberQ[{ArcSin, ...}, F] or
        # EqQ[F, Sin]) -> emit HeadRef(sympy.asin): a HeadRef is the same kind of object a
        # wildcard function head binds to (a Symbol subclass carrying the SymPy class), so
        # the test compares against the bound head instead of an unrelated Mathematica-named
        # Symbol. (The bare class itself is not an Expr and cannot sit in a constraint's args.)
        if atom in self._HEAD_FUNCTION_NAMES and atom in self.func_map:
            return f"HeadRef({self.func_map[atom]})"
        # Known wildcard references.
        #
        # ONLY outside a pattern. Inside a pattern a bare atom is a LITERAL symbol, and
        # Mathematica keeps it independent of a same-named pattern variable:
        # `MatchQ[d + 5 W, d + d_.*W]` is True -- the literal `d` matches itself while
        # the variable binds 5. Resolving bare atoms here unconditionally made the
        # translation ORDER-DEPENDENT, because these sets fill up as conversion walks the
        # tree: `d + d_.*W` came out correctly as `Symbol('d') + _d_*W`, but the very same
        # Mathematica expression written `d_.*W + d` (Plus is orderless, so they are the
        # SAME expression) came out as `_d_*W + _d_` -- which then demands both be equal
        # and no longer matches `d + 5 W`.
        #
        # On the REPLACEMENT / CONSTRAINT side there are no `Pattern[...]` nodes at all --
        # wildcards appear as bare atoms and MUST resolve to the bound values. That side
        # is converted with `is_pattern=False` (and the caller pre-seeds the names via
        # `wildcards=` / `optional_wildcards=`), so it is unaffected.
        if not is_pattern:
            if atom in self._wildcards_optional:
                var_name = f'_{atom}_'
                if var_name not in self._eval_ns:
                    ws = WildSymbol(atom, optional_value=IDENTITY_ELEMENT)
                    self._eval_ns[var_name] = ws
                return var_name
            if atom in self._wildcards_non_optional:
                var_name = f'{atom}_'
                if var_name not in self._eval_ns:
                    ws = WildSymbol(atom)
                    self._eval_ns[var_name] = ws
                return var_name
        # A scoping local (declared in an enclosing Module/With/Block binding list) is
        # emitted BARE -- it is declared once at the top of the generated module, so
        # bindings read `Module({r: ...}, ...)` instead of building `Symbol('r')`
        # inline. Any other plain symbol keeps the inline constructor (bare-ifying all
        # atoms would shadow single-letter imports like S/I/E). A local whose name is
        # also a KNOWN CALLABLE (func_map / custom_functions, e.g. the derivative `D`)
        # is NOT bare-ified either -- declaring `D = Symbol('D')` would shadow the
        # function the rule also calls. Only the actually-bare-ified names go into
        # ``_bare_locals`` (that is what the generator declares).
        self._symbols.add(atom)
        self._eval_ns.setdefault(atom, Symbol(atom))
        if (atom in self._scope_locals and atom.isidentifier()
                and not keyword.iskeyword(atom)
                and atom not in self.func_map and atom not in self._custom_functions):
            self._bare_locals.add(atom)
            return atom
        return f"Symbol('{atom}')"

    def _pattern_to_code(self, ffl) -> str:
        """['Pattern', name, ['Blank', ...]] -> wildcard reference."""
        name = ffl[1]
        if name in self._reserved_symbols:
            return self._reserved_symbols[name]
        self._wildcards_non_optional.add(name)
        var_name = f'{name}_'
        def_str = f"{var_name} = WildSymbol('{name}')"
        self._wild_defs.append(def_str)
        ws = WildSymbol(name)
        self._eval_ns[var_name] = ws
        return var_name

    def _optional_to_code(self, ffl) -> str:
        """['Optional', ['Pattern', name, ['Blank']]] -> optional wildcard."""
        inner = ffl[1]
        if isinstance(inner, list) and inner[0] == 'Pattern':
            name = inner[1]
            if name in self._reserved_symbols:
                # ``x_.`` where ``x`` is ALSO bound externally (e.g. by ``x_Symbol``).
                # Both bind the same name, so the "absent" branch would have to give
                # ``x`` the Times identity 1 -- which then fails ``x_Symbol``. The
                # optional branch is therefore unreachable and the factor is in fact
                # mandatory. Verified in Mathematica: ``g[x_.*h[x_], x_Symbol]``
                # matches ``g[z h[z], z]`` but NOT ``g[h[z], z]``.
                return self._reserved_symbols[name]
            self._wildcards_non_optional.add(name)
            self._wildcards_optional.add(name)
            var_name = f'_{name}_'
            def_str = f"{var_name} = WildSymbol('{name}', optional_value=IDENTITY_ELEMENT)"
            self._wild_defs.append(def_str)
            ws = WildSymbol(name, optional_value=IDENTITY_ELEMENT)
            self._eval_ns[var_name] = ws
            return var_name
        return self.convert(inner, is_pattern=True)

    def _function_to_code(self, ffl, *, is_pattern: bool = False) -> str:
        """Convert Mathematica Function[...] to sympy.Lambda(...).

        Supported forms:
          ["Function", body]             -> Lambda(Symbol('xi1'), body)
            where Slot[1] in body is replaced with Symbol('xi1')
          ["Function", param, body]      -> Lambda(Symbol(param), body)
          ["Function", ["List", ...], body] -> Lambda(tuple_of_symbols, body)
        """
        if len(ffl) == 2:
            # Pure function with Slot references: Function[body]
            body = ffl[1]
            # Find all Slot numbers used in body
            slot_nums = self._collect_slots(body)
            if not slot_nums:
                slot_nums = {'1'}  # default single arg
            # Build slot variable mapping
            old_slot_vars = self._slot_vars.copy()
            self._slot_vars = {n: f"Symbol('xi{n}')" for n in slot_nums}
            try:
                body_code = self.convert(body, is_pattern=is_pattern)
            finally:
                self._slot_vars = old_slot_vars
            # Build Lambda
            sorted_nums = sorted(slot_nums, key=int)
            if len(sorted_nums) == 1:
                param_code = f"Symbol('xi{sorted_nums[0]}')"
            else:
                params = ', '.join(f"Symbol('xi{n}')" for n in sorted_nums)
                param_code = f"({params})"
            return f"Lambda({param_code}, {body_code})"

        elif len(ffl) == 3:
            # Named parameter form: Function[param, body] or Function[{params}, body]
            params_ffl = ffl[1]
            body = ffl[2]
            if isinstance(params_ffl, str):
                # Single named parameter
                old_slot_vars = self._slot_vars.copy()
                self._slot_vars = {}  # no Slot mapping needed
                try:
                    body_code = self.convert(body, is_pattern=is_pattern)
                finally:
                    self._slot_vars = old_slot_vars
                return f"Lambda(Symbol('{params_ffl}'), {body_code})"
            elif isinstance(params_ffl, list) and params_ffl[0] == 'List':
                # Multiple named parameters: Function[{x, y}, body]
                param_names = params_ffl[1:]
                old_slot_vars = self._slot_vars.copy()
                self._slot_vars = {}
                try:
                    body_code = self.convert(body, is_pattern=is_pattern)
                finally:
                    self._slot_vars = old_slot_vars
                params = ', '.join(f"Symbol('{p}')" for p in param_names)
                return f"Lambda(({params}), {body_code})"

        # Fallback — shouldn't normally be reached
        args = [self.convert(a, is_pattern=is_pattern) for a in ffl[1:]]
        return f"Lambda({', '.join(args)})"

    def _derivative_to_code(self, head, args, is_pattern: bool) -> Optional[str]:  # noqa: C901
        """Code for ``Derivative[n1, ...][f][args]``, or None if *head* is not one."""
        if not (isinstance(head, list) and len(head) == 2
                and isinstance(head[0], list) and head[0] and head[0][0] == 'Derivative'
                and isinstance(head[1], str)):
            return None
        orders = head[0][1:]
        if len(orders) != len(args) or not all(
                isinstance(o, str) and o.isdigit() for o in orders):
            return None
        arg_codes = [self.convert(a, is_pattern=is_pattern) for a in args]
        if not any(int(o) for o in orders):
            return f"{self.convert([head[1]] + list(args), is_pattern=is_pattern)}"
        if all(isinstance(a, str) and not _is_numeric_atom(a) for a in args):
            # f'[x]: differentiate with respect to the symbol itself.
            func = self.convert([head[1]] + list(args), is_pattern=is_pattern)
            spec = [f"({a}, {int(o)})" for a, o in zip(arg_codes, orders) if int(o) > 0]
            return f"sympy.Derivative({func}, {', '.join(spec)})"
        # f'[Sin[x]]: the derivative of f evaluated AT sin(x). SymPy cannot
        # differentiate with respect to sin(x); the faithful form is the one its own
        # diff() produces, Subs(Derivative(f(_xi_1), _xi_1), _xi_1, sin(x)). The
        # dummies are plain Symbols (a Dummy would not round-trip through printing)
        # registered like any other symbol so the shortening namespace knows them.
        dummies = []
        for i in range(len(args)):
            name = f'_xi_{i + 1}'
            self._symbols.add(name)
            self._eval_ns.setdefault(name, Symbol(name))
            dummies.append(f"Symbol('{name}')")
        func = f"sympy.Function('{head[1]}')({', '.join(dummies)})"
        spec = [f"({d}, {int(o)})" for d, o in zip(dummies, orders) if int(o) > 0]
        deriv = f"sympy.Derivative({func}, {', '.join(spec)})"
        if len(args) == 1:
            return f"sympy.Subs({deriv}, {dummies[0]}, {arg_codes[0]})"
        return f"sympy.Subs({deriv}, ({', '.join(dummies)}), ({', '.join(arg_codes)}))"

    @staticmethod
    def _collect_slots(ffl) -> Set[str]:
        """Recursively collect all Slot numbers from an FFL subtree."""
        result: Set[str] = set()
        if isinstance(ffl, list) and len(ffl) > 0:
            if ffl[0] == 'Slot':
                result.add(ffl[1] if len(ffl) > 1 else '1')
            else:
                for child in ffl[1:]:
                    result.update(FFLConverter._collect_slots(child))
        return result


# ---------------------------------------------------------------------------
# FFL-level API (skip Mathematica parsing)
# ---------------------------------------------------------------------------


def ffl_to_sympy_code(
    ffl: Any,
    reserved_symbols: Optional[Mapping[str, str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    custom_functions: Optional[CustomFunctionsDict] = None,
    wildcards: Optional[Set[str]] = None,
    optional_wildcards: Optional[Set[str]] = None,
) -> Tuple[str, List[str], list[str]]:
    """Convert a Full-Form List to an eval-able Python code string.

    This is the lower-level entry point that operates directly on an FFL
    structure, skipping the Mathematica string-parsing step.

    Parameters
    ----------
    ffl : list or str
        A Full-Form List (nested list), e.g. ``['Power', 'x', '2']``.
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names bound externally by
        the caller (never turned into pattern wildcards).
    custom_functions : dict, optional
        Mapping from Wolfram head names to custom callables.  Each value is
        a 2-tuple ``(qualified_code_str, obj)``—see
        :class:`~.ffl_to_sympy.FFLConverter` for details.
    wildcards : set of str, optional
        Pre-known non-optional wildcard names.  Bare atoms matching these
        names will be emitted as ``name_`` references (useful when processing
        replacement/constraint FFL where wildcards appear as plain atoms
        rather than ``['Pattern', ...]`` nodes).
    optional_wildcards : set of str, optional
        Pre-known optional wildcard names.  Bare atoms matching these names
        will be emitted as ``_name_`` references.

    namespace : dict, optional
        The namespace the emitted code can be evaluated in. UPDATED IN PLACE with
        the base SymPy names, the reserved symbols and every wildcard discovered
        during conversion. Pass your own dict to add names of your own; they take
        precedence over the base names.

    Returns
    -------
    code : str
        A Python expression string that evaluates to a SymPy ``Basic`` object.
    wild_defs : list of str
        Variable definition statements for WildSymbol declarations.
    symbols : list of str
        Names of the plain symbols encountered.

    Examples
    --------
    >>> ns = {}
    >>> code, defs, symbols = ffl_to_sympy_code(['Power', 'x', '2'], {'x': 'x'}, ns)
    >>> code
    '(x)**(Integer(2))'
    >>> eval(code, ns)
    x**2
    """
    converter = FFLConverter(reserved_symbols=reserved_symbols,
                             custom_functions=custom_functions)
    if wildcards:
        converter._wildcards_non_optional.update(wildcards)
    if optional_wildcards:
        converter._wildcards_optional.update(optional_wildcards)
        converter._wildcards_non_optional.update(optional_wildcards)

    # The caller's namespace is filled in place, in order of increasing priority:
    #   base SymPy names  <  reserved symbols  <  the caller's own entries
    # and then, during convert(), every wildcard the conversion discovers.
    # A reserved symbol MUST beat the base names: the base binds ``x`` to
    # Symbol('x'), but a rule written over ``t`` maps 't' -> identifier 'x' and
    # needs that identifier to evaluate to Symbol('t').
    if namespace is None:
        namespace = {}
    caller_entries = dict(namespace)
    converter.use_namespace(namespace)
    for wolfram_name, identifier in (reserved_symbols or {}).items():
        namespace[identifier] = Symbol(wolfram_name)
    namespace.update(caller_entries)

    # A conversion with NO caller-supplied wildcard names is a PATTERN: its wildcards
    # arrive as `Pattern[...]`/`Optional[...]` nodes, and any BARE atom is a LITERAL
    # symbol that Mathematica keeps independent of a same-named pattern variable
    # (`MatchQ[d + 5 W, d + d_.*W]` is True). Passing `is_pattern` through is what stops
    # a literal being rewritten into the wildcard, which used to make the translation
    # ORDER-DEPENDENT -- `d + d_.*W` converted correctly while the identical (Plus is
    # orderless) `d_.*W + d` did not.
    #
    # A conversion WITH caller-supplied names is the replacement/constraint side, where
    # wildcards legitimately appear as bare atoms and must resolve to the bound values.
    is_pattern = not (wildcards or optional_wildcards)
    code = converter.convert(ffl, is_pattern=is_pattern)
    return code, converter.wild_defs, sorted(converter._symbols)


def _simplify_code(code: str, ns: Dict[str, Any],
                   str_printer: Optional[StrPrinter] = None,
                   rewrite: Optional[Any] = None) -> str:
    """Shorten *code* by round-tripping it through a printer, when that is safe.

    ``eval`` the code, print the resulting object with *str_printer*, and keep the
    printed form only if evaluating it back in *ns* reproduces the same object --
    so the shortening can never change meaning. Otherwise return *code* unchanged.

    *str_printer* defaults to :data:`_wild_printer`, which prints WildSymbol with
    the ``m_`` / ``_m_`` variable-name convention rather than the bare SymPy name.
    Pass any other :class:`~sympy.printing.str.StrPrinter` subclass to control the
    emitted style.

    The equality test also accepts a symbolic match (``(recovered - obj).simplify()
    == 0``), not just structural equality, which shortens a few forms that are
    equal but not identical.
    """
    printer = str_printer if str_printer is not None else _wild_printer
    try:
        obj = eval(code, ns)
        if rewrite is not None:
            # A deliberate TRANSLATION applied before printing (e.g. Wolfram nodes ->
            # standard SymPy). It must happen here, not after: the round-trip below
            # verifies the printed text against `obj`, so rewriting afterwards would
            # always compare unequal and silently discard the result.
            obj = rewrite(obj)
        # An undefined function the namespace does not know (a Rubi marker such as
        # Unintegrable[...], an arbitrary F[x]) prints bare as ``Unintegrable(...)``,
        # which the round-trip cannot evaluate -- so the WHOLE expression used to
        # stay verbose, and the rewrite above was silently lost with it. Print those
        # heads qualified instead; placeholders the caller registered (FreeQ, ...)
        # are in the namespace and keep their bare call-form.
        if isinstance(obj, sympy.Basic) and isinstance(printer, _WildStrPrinter):
            unknown = frozenset(type(f).__name__ for f in obj.atoms(AppliedUndef)
                                if type(f).__name__ not in ns)
            # Likewise a plain Symbol whose name the namespace binds to something
            # else (a coefficient ``D`` vs the Wolfram ``D`` node).
            shadowed = frozenset(
                s_.name for s_ in obj.atoms(Symbol)
                if type(s_) is Symbol and s_.name in ns and ns[s_.name] != s_)
            if unknown or shadowed:
                printer = type(printer)()
                printer._qualify_undefined = unknown
                printer._qualify_symbols = shadowed
        short = printer.doprint(obj)
        recovered = eval(short, ns)
        if isinstance(recovered, sympy.Basic):
            if recovered == obj:
                return short
            if (recovered - obj).simplify() == 0:
                return short
    except Exception:
        pass
    return code


def ffl_to_sympy_short_code(
    ffl: Any,
    reserved_symbols: Optional[Mapping[str, str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    custom_functions: Optional[CustomFunctionsDict] = None,
    wildcards: Optional[Set[str]] = None,
    optional_wildcards: Optional[Set[str]] = None,
    str_printer: Optional[StrPrinter] = None,
    rewrite: Optional[Any] = None,
) -> Tuple[str, List[str], list[str]]:
    """Like :func:`ffl_to_sympy_code` but with a simplification pass.

    *rewrite*, when given, is a callable applied to the evaluated expression before it
    is printed -- used by callers that want a deliberate translation baked into the
    emitted code, e.g. ``sympy_wolfram.objects.rewrite_as_standard_sympy`` to turn
    Wolfram nodes into their standard SymPy equivalents.

    Operates directly on an FFL structure, skipping Mathematica parsing.

    Parameters
    ----------
    ffl : list or str
        A Full-Form List (nested list).
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names bound externally by
        the caller (never turned into pattern wildcards).
    namespace : dict, optional
        The namespace the emitted code is verified against. UPDATED IN PLACE with
        the base SymPy names, the reserved symbols and every wildcard discovered
        during conversion, so afterwards it can ``eval`` the returned code. Pass
        your own dict to make additional names visible to the round-trip.
    custom_functions : dict, optional
        Mapping from Wolfram head names to custom callables.
    wildcards : set of str, optional
        Pre-known non-optional wildcard names (see :func:`ffl_to_sympy_code`).
    optional_wildcards : set of str, optional
        Pre-known optional wildcard names (see :func:`ffl_to_sympy_code`).
    str_printer : StrPrinter, optional
        Printer used for the shortening round-trip. Defaults to a printer that
        renders WildSymbol as ``m_`` / ``_m_``; pass any StrPrinter subclass to
        control the emitted style. See :func:`_simplify_code`.

    Returns
    -------
    short_code : str
        A (possibly simplified) Python expression string.
    wild_defs : list of str
        WildSymbol variable definitions.
    symbols : list of str
        Names of the plain symbols encountered.

    Examples
    --------
    >>> ns = {}
    >>> short, defs, symbols = ffl_to_sympy_short_code(['Power', 'x', '2'], {'x': 'x'}, ns)
    >>> short
    'x**2'
    >>> eval(short, ns)
    x**2
    """
    if namespace is None:
        namespace = {}
    code, wild_defs, symbols = ffl_to_sympy_code(
        ffl, reserved_symbols, namespace, custom_functions=custom_functions,
        wildcards=wildcards, optional_wildcards=optional_wildcards,
    )
    return _simplify_code(code, namespace, str_printer, rewrite), wild_defs, symbols


# ---------------------------------------------------------------------------
# Mathematica-string-level API (parse + delegate to FFL API)
# ---------------------------------------------------------------------------


def mathematica_to_sympy_code(
    expr_str: str,
    reserved_symbols: Optional[Mapping[str, str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    custom_functions: Optional[CustomFunctionsDict] = None,
) -> Tuple[str, List[str], list[str]]:
    """Convert a Mathematica expression string to an eval-able Python code string.

    Pipeline: Mathematica notation → FFL → SymPy code string.
    Equivalent to ``ffl_to_sympy_code(mathematica_to_ffl(expr_str), ...)``.

    Parameters
    ----------
    expr_str : str
        Mathematica expression in standard notation.
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names bound externally by
        the caller (never turned into pattern wildcards).
    custom_functions : dict, optional
        Mapping from Wolfram head names to custom callables.  Each value is
        a 2-tuple ``(qualified_code_str, obj)``—see
        :class:`~.ffl_to_sympy.FFLConverter` for details.

    Returns
    -------
    code : str
        A Python expression string that evaluates to a SymPy ``Basic`` object.
        Uses ``sympy.*`` qualified names, ``Integer(...)``, ``Rational(...)``,
        ``Symbol(...)`` etc.  Pattern variables are referenced by name
        (e.g. ``m_`` or ``_m_``) and defined in *wild_defs*.
    eval_ns : dict
        A namespace dictionary suitable for passing to ``eval(code, eval_ns)``.
        Contains ``sympy``, common functions, a ``Symbol`` for each reserved
        name, and any WildSymbol variables.
    wild_defs : list of str
        Variable definition statements for WildSymbol declarations.  Each
        entry is a Python assignment string (e.g.
        ``"m_ = WildSymbol('m')"``) that should be exec'd before eval'ing
        *code* if building a standalone script.  The eval_ns already contains
        these bindings, so they are informational for code-generation use.

    Examples
    --------
    >>> code, ns, defs, _symbols = mathematica_to_sympy_code("Sin[x] + x^2")
    >>> code
    '(sympy.sin(x) + (x)**(Integer(2)))'
    >>> eval(code, ns)
    x**2 + sin(x)

    >>> code, ns, defs, _symbols = mathematica_to_sympy_code("m_")
    >>> code
    'm_'
    >>> defs
    ["m_ = WildSymbol('m')"]
    """
    ffl = mathematica_to_ffl(expr_str)
    return ffl_to_sympy_code(ffl, reserved_symbols, namespace,
                             custom_functions=custom_functions)


def mathematica_to_sympy_short_code(
    expr_str: str,
    reserved_symbols: Optional[Mapping[str, str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    custom_functions: Optional[CustomFunctionsDict] = None,
    str_printer: Optional[StrPrinter] = None,
) -> Tuple[str, List[str], list[str]]:
    """Like :func:`mathematica_to_sympy_code` but with a simplification pass.

    Pipeline: Mathematica notation → FFL → simplified code string.
    Equivalent to ``ffl_to_sympy_short_code(mathematica_to_ffl(expr_str), ...)``.

    Parameters
    ----------
    expr_str : str
        Mathematica expression in standard notation.
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names bound externally by
        the caller (never turned into pattern wildcards).
    namespace : dict, optional
        Namespace the code is verified against; updated in place (see
        :func:`ffl_to_sympy_short_code`). Pre-populate it with any free
        parameters the expression mentions.
    custom_functions : dict, optional
        Mapping from Wolfram head names to custom callables.  See
        :class:`~.ffl_to_sympy.FFLConverter` for details.

    Returns
    -------
    short_code : str
        A (possibly simplified) Python expression string.
    wild_defs : list of str
        WildSymbol variable definitions (same as from
        :func:`mathematica_to_sympy_code`).
    symbols : list of str
        Names of the plain symbols encountered.

    Examples
    --------
    >>> ns = {}
    >>> short, defs, symbols = mathematica_to_sympy_short_code(
    ...     "Sin[x] + x^2", {'x': 'x'}, ns)
    >>> short
    'x**2 + sin(x)'
    >>> eval(short, ns)
    x**2 + sin(x)

    >>> ns = {'a': Symbol('a'), 'b': Symbol('b'), 'm': Symbol('m')}
    >>> short, defs, symbols = mathematica_to_sympy_short_code(
    ...     "(a + b*x)^m", {'x': 'x'}, ns)
    >>> short
    '(a + b*x)**m'
    """
    ffl = mathematica_to_ffl(expr_str)
    return ffl_to_sympy_short_code(
        ffl, reserved_symbols, namespace,
        custom_functions=custom_functions, str_printer=str_printer,
    )


def mathematica_to_sympy(
    expr_str: str,
    reserved_symbols: Optional[Mapping[str, str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    custom_functions: Optional[CustomFunctionsDict] = None,
) -> sympy.Basic:
    """Convert a Mathematica expression string directly to a SymPy object.

    Pipeline: Mathematica notation → FFL → SymPy code string → eval.

    Parameters
    ----------
    expr_str : str
        Mathematica expression in standard notation.
    reserved_symbols : mapping of str to str, optional
        Wolfram symbol name -> Python identifier, for names bound externally by
        the caller (never turned into pattern wildcards).
    namespace : dict, optional
        Namespace the code is evaluated in; updated in place. Pre-populate it so
        free parameters evaluate to the SymPy symbols you intend.
    custom_functions : dict, optional
        Mapping from Wolfram head names to custom callables.  See
        :class:`~.ffl_to_sympy.FFLConverter` for details.

    Returns
    -------
    sympy.Basic
        The resulting SymPy expression.

    Examples
    --------
    >>> from sympy import Symbol, sin
    >>> mathematica_to_sympy("Sin[x]") == sin(Symbol('x'))
    True
    >>> a, b, x = Symbol('a'), Symbol('b'), Symbol('x')
    >>> mathematica_to_sympy("(a + b*x)^2", namespace={'a': a, 'b': b})
    (a + b*x)**2
    """
    if namespace is None:
        namespace = {}
    code, _wild_defs, _symbols = mathematica_to_sympy_code(
        expr_str, reserved_symbols, namespace, custom_functions=custom_functions
    )
    return eval(code, namespace)

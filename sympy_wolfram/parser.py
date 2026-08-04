# -*- coding: utf-8 -*-
"""Parser: Wolfram Mathematica source text -> Full-Form List (FFL).

This is the FRONT of the pipeline and its only job is syntax: turn Mathematica
standard notation into the nested-list FFL representation. It assigns no
meaning -- every node stays a plain string head with plain arguments.

    parser.py       text     -> FFL      (this module)
    interpreter.py  FFL      -> SymPy    (meaning is assigned there)
    objects.py      the Mathematica objects the interpreter can emit
"""
from __future__ import annotations

from typing import List

# ``parse_mathematica_to_fullformlist`` is not part of released sympy (it
# exists in sympy 1.15.0.dev); guard the import so this package works on stock
# sympy -- the FFL parser is only needed at rule-GENERATION time, never for
# integration, so the failure is deferred to the actual call with a clear
# message instead of breaking ``import sympy_wolfram`` for everyone.
try:
    from sympy.parsing.mathematica import parse_mathematica_to_fullformlist
except ImportError:  # pragma: no cover -- depends on installed sympy version
    parse_mathematica_to_fullformlist = None


def mathematica_to_ffl(expr_str: str) -> List:
    """Convert a Mathematica standard-notation string to a Full-Form List.

    Parameters
    ----------
    expr_str : str
        A Mathematica expression in standard notation, e.g.
        ``"(a + b*x)^m"``, ``"Sin[x] + Cos[x]"``, ``"Log[1 + x]/x"``.

    Returns
    -------
    list
        Nested list in Full-Form List format, e.g.
        ``['Power', ['Plus', 'a', ['Times', 'b', 'x']], 'm']``.

    Examples
    --------
    >>> mathematica_to_ffl("x^2")
    ['Power', 'x', '2']
    >>> mathematica_to_ffl("Sin[x] + 1")
    ['Plus', ['Sin', 'x'], '1']
    """
    if parse_mathematica_to_fullformlist is None:
        raise RuntimeError(
            'mathematica_to_ffl requires sympy.parsing.mathematica.'
            'parse_mathematica_to_fullformlist, which this sympy version '
            'does not provide (it exists in sympy >= 1.15.0.dev). This is '
            'only needed to parse Mathematica sources, not for integration.')
    return parse_mathematica_to_fullformlist(expr_str)

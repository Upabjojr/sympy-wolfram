# -*- coding: utf-8 -*-
"""Execute every doctest in sympy_wolfram's Markdown documentation (README + docs/).

Documentation that is not executed drifts silently from the code; these documents
describe behaviour where a divergence produces wrong results rather than errors, so
every example runs as a test.
"""
import doctest
import pathlib
import warnings

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent
DOCS = sorted([PKG / 'README.md'] + list((PKG / 'docs').glob('*.md')))


# Docs whose examples parse Mathematica SOURCE STRINGS, which needs sympy's
# parse_mathematica_to_fullformlist — absent from released sympy (<= 1.14).
PARSER_DOCS = {'translating-mathematica.md'}


@pytest.mark.parametrize('doc', DOCS, ids=lambda p: p.name)
def test_doc_examples(doc):
    if doc.name in PARSER_DOCS:
        from sympy_wolfram.parser import parse_mathematica_to_fullformlist
        if parse_mathematica_to_fullformlist is None:
            pytest.skip('needs sympy.parsing.mathematica.parse_mathematica_to_'
                        'fullformlist (sympy > 1.14)')
    warnings.filterwarnings('ignore')
    result = doctest.testfile(
        str(doc),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert result.attempted > 0, f'no doctests collected from {doc.name}'
    assert result.failed == 0, f'{result.failed} of {result.attempted} examples failed in {doc.name}'

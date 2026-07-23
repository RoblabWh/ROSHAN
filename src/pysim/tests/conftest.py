"""Test path setup.

ppo.py (and its transitive imports) use ``src/pysim``-relative module names
(``from algorithms... import``, ``from utils import``) and the compiled ``firesim``
module, exactly as when ``main.py`` is launched with ``python src/pysim/main.py``.
Put both directories on sys.path so the real modules import under pytest.
"""
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))  # tests -> pysim -> src -> root
_PYSIM = os.path.join(_ROOT, "src", "pysim")
_BUILD = os.path.join(_ROOT, "build")

for _p in (_PYSIM, _BUILD):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

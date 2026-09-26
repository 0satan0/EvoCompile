"""Multi-agent compile loop: collect → retrieve → LLM recipe → eval → repair."""

from pathlib import Path
import sys

GKO = Path(__file__).resolve().parents[1]
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)
if str(GKO) not in sys.path:
    sys.path.insert(0, str(GKO))

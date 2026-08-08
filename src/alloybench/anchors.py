"""Validation anchors: real DFT from the Materials Project, plus cited
experimental values where MP has none.

Anchors are the ONLY non-model reference numbers in this project. The
Materials Project values (CC BY 4.0) are fetched by
scripts/fetch_mp_anchors.py into data/anchors_mp.json; if that file is
absent, the committed literature fallback data/anchors_literature.json is
used. Training labels are surrogate teacher labels and never appear here.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_anchors(data_dir: str | Path = "data") -> dict:
    """Load MP anchors if fetched, else the committed literature fallback.

    Returns:
        The anchor payload with an added "which" key ("mp" or "literature").
    """
    data_dir = Path(data_dir)
    mp = data_dir / "anchors_mp.json"
    lit = data_dir / "anchors_literature.json"
    if mp.exists():
        payload = json.loads(mp.read_text())
        payload["which"] = "mp"
        # Experimental values (elastic constants fallback, thermal expansion,
        # vacancy range) always come from the literature file.
        if lit.exists():
            payload.setdefault("experimental", json.loads(lit.read_text()).get("experimental", {}))
        return payload
    payload = json.loads(lit.read_text())
    payload["which"] = "literature"
    return payload

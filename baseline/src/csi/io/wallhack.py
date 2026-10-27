"""Wallhack1.8k raw files: <root>/{LOS,NLOS}/{BQ,PIFA}/{b1,w1..w5,ww1..ww5}.csv."""
from __future__ import annotations

import re
from pathlib import Path

DOMAINS = [("LOS", "BQ"), ("LOS", "PIFA"), ("NLOS", "BQ"), ("NLOS", "PIFA")]


def list_recordings(root) -> list[dict]:
    """One dict per raw CSV with the metadata known from the file layout."""
    root = Path(root)
    recs = []
    for condition, antenna in DOMAINS:
        for path in sorted((root / condition / antenna).glob("*.csv")):
            m = re.fullmatch(r"(b|w|ww)(\d)", path.stem)
            if m is None:
                raise ValueError(f"unexpected file {path}")
            kind, num = m.group(1), int(m.group(2))
            # readme.txt: rooms 1..5 only for NLOS activity files; 0 = no presence
            room = 0 if kind == "b" else (num if condition == "NLOS" else None)
            recs.append({
                "recording_id": f"wallhack_{condition}_{antenna}_{path.stem}",
                "dataset": "wallhack",
                "condition": condition,
                "antenna": antenna,
                "day": None,
                "room": room,
                "session": path.stem,
                "path": path,
            })
    return recs

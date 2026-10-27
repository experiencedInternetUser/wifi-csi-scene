"""3DO raw files: <root>/d{1,2,3}/{b,w,s,l}<n>/csiposreg.csv.

`d1/b*/csi.csv` is the same stream without x,y,z and is ignored.
"""
from __future__ import annotations

from pathlib import Path


def list_recordings(root) -> list[dict]:
    root = Path(root)
    recs = []
    for path in sorted(root.glob("d*/*/csiposreg.csv")):
        day, session = path.parent.parent.name, path.parent.name
        recs.append({
            "recording_id": f"3do_{day}_{session}",
            "dataset": "3do",
            "condition": "through-wall",
            "antenna": None,  # not stated in the data; see docs/DATA.md
            "day": day,
            "room": None,
            "session": session,
            "path": path,
        })
    return recs

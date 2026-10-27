"""
store.py - per-user library of uploaded scans and saved views, kept on the server's disk.

Layout:  <BEAMLAB_DATA>/<user>/index.json
         <BEAMLAB_DATA>/<user>/scans/<content-id>.mat

BEAMLAB_DATA defaults to ./beamlab_data next to the app. On a cloud host, point it at a
persistent volume (otherwise uploads disappear when the container restarts).
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from beamcore import Scan, content_id, is_scan_file


def _slug(user: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.@-]+", "_", user.strip().lower()) or "local"


class UserStore:
    def __init__(self, user: str, root: Optional[str] = None):
        base = Path(root or os.environ.get("BEAMLAB_DATA") or Path(__file__).with_name("beamlab_data"))
        self.dir = base / _slug(user)
        self.scan_dir = self.dir / "scans"
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.data = self._read()

    # ------------------------------------------------------------------ io
    def _read(self) -> dict:
        try:
            d = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        d.setdefault("scans", [])
        d.setdefault("views", [])
        return d

    def _save(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def reload(self) -> None:
        self.data = self._read()

    # ------------------------------------------------------------------ scans
    @property
    def scans(self) -> list:
        return self.data["scans"]

    def scan(self, sid: str) -> Optional[dict]:
        return next((s for s in self.scans if s["id"] == sid), None)

    def scan_path(self, sid: str) -> Path:
        return self.scan_dir / f"{sid}.mat"

    def add_scan(self, data: bytes, filename: str) -> dict:
        """Store an uploaded scan (deduplicated by content). Raises ValueError if it isn't a scan."""
        if not is_scan_file(data):
            raise ValueError(f"{filename} isn't a beam-scan .mat file (no VPPDatastore).")
        sid = content_id(data)
        self.reload()
        existing = self.scan(sid)
        if existing:
            self.scans.remove(existing)
            self.scans.insert(0, existing)
            self._save()
            return existing
        sc = Scan.load(data, filename)                       # validates + gives metadata
        self.scan_path(sid).write_bytes(data)
        entry = dict(id=sid, name=sc.title, filename=filename, plane=sc.plane, freq=sc.freq,
                     added=time.strftime("%Y-%m-%d %H:%M"))
        self.scans.insert(0, entry)
        self._save()
        return entry

    def remove_scan(self, sid: str) -> None:
        self.reload()
        e = self.scan(sid)
        if e:
            self.scans.remove(e)
            try:
                self.scan_path(sid).unlink()
            except OSError:
                pass
            self._save()

    # ------------------------------------------------------------------ views
    @property
    def views(self) -> list:
        return self.data["views"]

    def view(self, vid: str) -> Optional[dict]:
        return next((v for v in self.views if v["id"] == vid), None)

    def add_view(self, name: str, scan_id: str, scan_name: str, settings: dict, display: dict) -> dict:
        self.reload()
        v = dict(id=uuid.uuid4().hex[:10], name=name, scan_id=scan_id, scan_name=scan_name,
                 created=time.strftime("%Y-%m-%d %H:%M"), settings=settings, display=display)
        self.views.append(v)
        self._save()
        return v

    def update_view(self, vid: str, **fields) -> None:
        self.reload()
        v = self.view(vid)
        if v:
            v.update(fields)
            self._save()

    def delete_view(self, vid: str) -> None:
        self.reload()
        v = self.view(vid)
        if v:
            self.views.remove(v)
            self._save()

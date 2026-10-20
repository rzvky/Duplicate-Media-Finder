"""Small SQLite cache so re-scanning the same folder only reads new or changed photos."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


class FeatureCache:
    def __init__(self, db_path: Path, feature_version: int, video_version: int = 0):
        self.db_path = Path(db_path)
        self.version = str(feature_version)
        self.video_version = str(video_version)
        self._pending = 0
        try:
            self._open()
        except sqlite3.DatabaseError:
            # damaged cache: start fresh, it only speeds things up
            try:
                os.remove(self.db_path)
            except OSError:
                pass
            self._open()

    def _open(self):
        self.db = sqlite3.connect(str(self.db_path), timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, "
            "sha TEXT, feat TEXT, emb BLOB, emb_model TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS videos (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, data TEXT)"
        )
        row = self.db.execute("SELECT v FROM meta WHERE k='version'").fetchone()
        if row is None or row[0] != self.version:
            self.db.execute("DELETE FROM files")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (self.version,))
        row = self.db.execute("SELECT v FROM meta WHERE k='video_version'").fetchone()
        if row is None or row[0] != self.video_version:
            self.db.execute("DELETE FROM videos")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('video_version', ?)", (self.video_version,))
        self.db.commit()

    def get(self, path: str, size: int, mtime_ns: int) -> Optional[dict]:
        row = self.db.execute(
            "SELECT size, mtime_ns, sha, feat, emb, emb_model FROM files WHERE path=?", (path,)
        ).fetchone()
        if not row or row[0] != size or row[1] != mtime_ns or not row[3]:
            return None
        out = {"sha": row[2], "feat": json.loads(row[3]), "emb": None, "emb_model": row[5]}
        if row[4] is not None:
            out["emb"] = np.frombuffer(row[4], dtype=np.float32).copy()
        return out

    def put(self, path: str, size: int, mtime_ns: int, sha: str, feat: dict):
        self.db.execute(
            "INSERT INTO files (path, size, mtime_ns, sha, feat, emb, emb_model) VALUES (?,?,?,?,?,NULL,NULL) "
            "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, sha=excluded.sha, "
            "feat=excluded.feat, emb=CASE WHEN files.sha=excluded.sha THEN files.emb ELSE NULL END, "
            "emb_model=CASE WHEN files.sha=excluded.sha THEN files.emb_model ELSE NULL END",
            (path, size, mtime_ns, sha, json.dumps(feat, separators=(",", ":"))),
        )
        self._tick()

    def put_emb(self, path: str, emb: np.ndarray, model: str):
        self.db.execute(
            "UPDATE files SET emb=?, emb_model=? WHERE path=?",
            (np.asarray(emb, dtype=np.float32).tobytes(), model, path),
        )
        self._tick()

    def vget(self, path: str, size: int, mtime_ns: int) -> Optional[dict]:
        row = self.db.execute("SELECT size, mtime_ns, data FROM videos WHERE path=?", (path,)).fetchone()
        if not row or row[0] != size or row[1] != mtime_ns or not row[2]:
            return None
        try:
            data = json.loads(row[2])
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def vput(self, path: str, size: int, mtime_ns: int, data: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO videos (path, size, mtime_ns, data) VALUES (?,?,?,?)",
            (path, size, mtime_ns, json.dumps(data, separators=(",", ":"))),
        )
        self._tick()

    def _tick(self):
        self._pending += 1
        if self._pending >= 200:
            self.commit()

    def commit(self):
        try:
            self.db.commit()
        except sqlite3.Error:
            pass
        self._pending = 0

    def close(self):
        self.commit()
        try:
            self.db.close()
        except sqlite3.Error:
            pass

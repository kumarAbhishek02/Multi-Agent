import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    status      TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    usage       TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    type   TEXT NOT NULL,
    data   TEXT NOT NULL,
    ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id     TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    content    TEXT NOT NULL,
    agent      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""

TERMINAL = ("completed", "failed", "cancelled")


class Store:
    """SQLite persistence for runs, their event logs and shared artifacts."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _run(row) -> dict:
        run = dict(row)
        run["usage"] = json.loads(run["usage"])
        return run

    # -- runs ---------------------------------------------------------------
    def create_run(self, run_id: str, task: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, task, status, created_at) VALUES (?, ?, 'queued', ?)",
                (run_id, task, time.time()),
            )

    def start_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status='running', started_at=? WHERE id=?", (time.time(), run_id)
            )

    def finish_run(self, run_id: str, status: str, result=None, error=None, usage=None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, result=?, error=?, usage=?, finished_at=? WHERE id=?",
                (status, result, error, json.dumps(usage or {}), time.time(), run_id),
            )

    def get_run(self, run_id: str):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run(row) if row else None

    def list_runs(self, limit: int = 50) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, task, status, usage, created_at, started_at, finished_at, "
                "NULL AS result, error FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._run(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def unfinished_runs(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM runs WHERE status IN ('queued', 'running')"
            ).fetchall()
        return [r["id"] for r in rows]

    # -- events -------------------------------------------------------------
    def add_event(self, run_id: str, type_: str, data: dict) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (run_id, type, data, ts) VALUES (?, ?, ?, ?)",
                (run_id, type_, json.dumps(data), time.time()),
            )
            return cur.lastrowid

    def events_after(self, run_id: str, after: int = 0) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT seq, type, data, ts FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                (run_id, after),
            ).fetchall()
        return [
            {"seq": r["seq"], "type": r["type"], "ts": r["ts"], **json.loads(r["data"])}
            for r in rows
        ]

    # -- artifacts ----------------------------------------------------------
    def save_artifact(self, run_id: str, name: str, content: str, agent: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO artifacts (run_id, name, content, agent, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id, name) DO UPDATE SET "
                "content=excluded.content, agent=excluded.agent, updated_at=excluded.updated_at",
                (run_id, name, content, agent, time.time()),
            )

    def get_artifact(self, run_id: str, name: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT name, content, agent, updated_at FROM artifacts WHERE run_id=? AND name=?",
                (run_id, name),
            ).fetchone()
        return dict(row) if row else None

    def list_artifacts(self, run_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, agent, updated_at, length(content) AS chars FROM artifacts "
                "WHERE run_id=? ORDER BY updated_at",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]


class EventHub:
    """Persists run events and wakes any SSE streams waiting for them."""

    def __init__(self, store: Store):
        self.store = store
        self._cond = threading.Condition()
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def emit(self, run_id: str, type_: str, data: dict) -> None:
        self.store.add_event(run_id, type_, data)
        with self._cond:
            self._version += 1
            self._cond.notify_all()

    def wait(self, seen_version: int, timeout: float) -> bool:
        """Block until something was emitted after `seen_version`, or timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._version != seen_version, timeout)

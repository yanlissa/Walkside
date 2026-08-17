import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


STATUS_OK = 1
STATUS_MISSING = 2


@dataclass(frozen=True)
class TileCacheEntry:
    status: str
    content: bytes | None = None


class SQLiteTileCache:

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        return connection

    def _initialize(self):
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    z INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    status INTEGER NOT NULL,
                    content BLOB,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (z, x, y),
                    CHECK (status IN (1, 2)),
                    CHECK (
                        (status = 1 AND content IS NOT NULL)
                        OR
                        (status = 2 AND content IS NULL)
                    )
                ) WITHOUT ROWID
                """
            )
        finally:
            connection.close()

    def _connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        return connection

    def get(self, z, x, y, missing_ttl_seconds):
        z = int(z)
        x = int(x)
        y = int(y)

        row = self._connection().execute(
            """
            SELECT status, content, updated_at
            FROM tiles
            WHERE z = ? AND x = ? AND y = ?
            """,
            (z, x, y),
        ).fetchone()

        if row is None:
            return None

        status, content, updated_at = row

        if status == STATUS_OK:
            return TileCacheEntry(
                status="ok",
                content=bytes(content),
            )

        if status == STATUS_MISSING:
            ttl = max(0.0, float(missing_ttl_seconds))
            age = time.time() - float(updated_at)

            if ttl > 0 and age <= ttl:
                return TileCacheEntry(status="missing")

            self.delete(z, x, y)
            return None

        # The CHECK constraint should make this impossible, but do not let a
        # damaged/old row poison processing forever.
        self.delete(z, x, y)
        return None

    def put_ok(self, z, x, y, content):
        self._connection().execute(
            """
            INSERT INTO tiles (z, x, y, status, content, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(z, x, y) DO UPDATE SET
                status = excluded.status,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (
                int(z),
                int(x),
                int(y),
                STATUS_OK,
                sqlite3.Binary(content),
                time.time(),
            ),
        )

    def put_missing(self, z, x, y):
        self._connection().execute(
            """
            INSERT INTO tiles (z, x, y, status, content, updated_at)
            VALUES (?, ?, ?, ?, NULL, ?)
            ON CONFLICT(z, x, y) DO UPDATE SET
                status = excluded.status,
                content = NULL,
                updated_at = excluded.updated_at
            """,
            (
                int(z),
                int(x),
                int(y),
                STATUS_MISSING,
                time.time(),
            ),
        )

    def delete(self, z, x, y):
        self._connection().execute(
            "DELETE FROM tiles WHERE z = ? AND x = ? AND y = ?",
            (int(z), int(x), int(y)),
        )

    def close_current_thread(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

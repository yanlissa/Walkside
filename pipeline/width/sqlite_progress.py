import hashlib
import sqlite3
import time
from pathlib import Path


WIDTH_PROGRESS_VERSION = 1


def _file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return f"{path.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def build_width_run_key(
    graph_path,
    weights_path,
    z,
    context_px,
    tile_size,
):
    value = "|".join([
        str(WIDTH_PROGRESS_VERSION),
        _file_signature(graph_path),
        _file_signature(weights_path),
        str(int(z)),
        str(int(context_px)),
        str(int(tile_size)),
    ])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SQLiteWidthProgress:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
        )
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS width_results (
                run_key TEXT NOT NULL,
                feature_idx INTEGER NOT NULL,
                part_idx INTEGER NOT NULL,
                edge_id TEXT,
                status TEXT NOT NULL,
                width_m REAL,
                width_px REAL,
                width_mode TEXT,
                crosswalk_ratio REAL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (run_key, feature_idx, part_idx)
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()

    def get(self, run_key, feature_idx, part_idx):
        row = self.connection.execute(
            """
            SELECT status, width_m, width_px, width_mode, crosswalk_ratio
            FROM width_results
            WHERE run_key = ? AND feature_idx = ? AND part_idx = ?
            """,
            (
                str(run_key),
                int(feature_idx),
                int(part_idx),
            ),
        ).fetchone()

        if row is None:
            return None

        return {
            "status": row[0],
            "width_m": row[1],
            "width_px": row[2],
            "width_mode": row[3],
            "crosswalk_ratio": row[4],
        }

    def put(self, run_key, edge, result):
        edge_id = edge.get("properties", {}).get("EdgeId")
        if edge_id is not None:
            edge_id = str(edge_id)

        self.connection.execute(
            """
            INSERT INTO width_results (
                run_key,
                feature_idx,
                part_idx,
                edge_id,
                status,
                width_m,
                width_px,
                width_mode,
                crosswalk_ratio,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key, feature_idx, part_idx) DO UPDATE SET
                edge_id = excluded.edge_id,
                status = excluded.status,
                width_m = excluded.width_m,
                width_px = excluded.width_px,
                width_mode = excluded.width_mode,
                crosswalk_ratio = excluded.crosswalk_ratio,
                updated_at = excluded.updated_at
            """,
            (
                str(run_key),
                int(edge["feature_idx"]),
                int(edge["part_idx"]),
                edge_id,
                str(result.get("status", "no_results")),
                _optional_float(result.get("width_m")),
                _optional_float(result.get("width_px")),
                result.get("width_mode"),
                _optional_float(result.get("crosswalk_ratio")),
                time.time(),
            ),
        )
        self.connection.commit()

    def count(self, run_key):
        row = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM width_results
            WHERE run_key = ?
            """,
            (str(run_key),),
        ).fetchone()
        return int(row[0])

    def close(self):
        self.connection.close()

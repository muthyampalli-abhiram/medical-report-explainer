"""SQLite database connection and schema management."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from app.config import settings


def get_db_path() -> Path:
    """Resolve the database file path from settings."""
    db_path = Path(settings.DB_PATH)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    return db_path


def init_db(db_path: Path | str | None = None) -> Path:
    """Create the storage directory, SQLite file, schema tables, and default profile if missing."""
    resolved_path = Path(db_path) if db_path is not None else get_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                report_date TEXT,
                lab_name TEXT,
                source_filename TEXT,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(profile_id, content_hash)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                report_id TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
                test_name_raw TEXT NOT NULL,
                test_name_canonical TEXT,
                value REAL,
                value_text TEXT,
                unit TEXT,
                ref_low REAL,
                ref_high REAL,
                ref_text TEXT,
                flag TEXT NOT NULL,
                value_original REAL,
                unit_original TEXT,
                source_page INTEGER,
                source_snippet TEXT,
                confidence REAL NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_results_canonical ON results(test_name_canonical);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reports_profile_date ON reports(profile_id, report_date);
            """
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO profiles (label, created_at) VALUES (?, ?);",
            ("Me", now_iso),
        )
        conn.commit()
    finally:
        conn.close()

    return resolved_path


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database with foreign keys enabled."""
    resolved_path = init_db(db_path)
    conn = sqlite3.connect(resolved_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

import sqlite3
from pathlib import Path
from threading import RLock

MAX_HISTORY = 10
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "chat_history.db"

_lock = RLock()

def _get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )
    conn.row_factory = sqlite3.Row
    # Cho phép đọc trong khi process khác đang ghi
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _init_database():
    with _lock:
        conn = _get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_chat_history_session
                ON chat_history(session_id, id)
            """)
            conn.commit()
        finally:
            conn.close()

def _normalize_session_id(session_id):
    session_id = str(
        session_id or ""
    ).strip()
    if not session_id:
        raise ValueError(
            "session_id không được để trống"
        )
    return session_id

def add_message(session_id, role, content):
    session_id = _normalize_session_id(
        session_id
    )
    role = str(role or "").strip()
    content = str(content or "").strip()
    if not role or not content:
        return
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chat_history(
                    session_id,
                    role,
                    content
                )
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content
                )
            )
            conn.execute(
                """
                DELETE FROM chat_history
                WHERE session_id = ?
                AND id NOT IN (
                    SELECT id
                    FROM chat_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (
                    session_id,
                    session_id,
                    MAX_HISTORY
                )
            )
            conn.commit()
        finally:
            conn.close()


def get_history(session_id):
    session_id = _normalize_session_id(
        session_id
    )
    with _lock:
        conn = _get_connection()
        try:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,)
            ).fetchall()
        finally:
            conn.close()
    return [
        {
            "role": row["role"],
            "content": row["content"]
        }
        for row in rows
    ]

def clear_history(session_id):
    session_id = _normalize_session_id(
        session_id
    )
    with _lock:
        conn = _get_connection()
        try:
            conn.execute(
                """
                DELETE FROM chat_history
                WHERE session_id = ?
                """,
                (session_id,)
            )
            conn.commit()
        finally:
            conn.close()
_init_database()
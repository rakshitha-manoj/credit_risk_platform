import sqlite3
import json
from datetime import datetime, timezone
from src.utils.config import config
from src.utils.logger import get_logger
logger = get_logger(__name__)
AUDIT_DB_PATH = config.SQL_DB_PATH

def _get_conn():
    return sqlite3.connect(AUDIT_DB_PATH)

def init_audit_tables() -> None:
    conn = _get_conn()
    conn.execute('\n        CREATE TABLE IF NOT EXISTS audit_predictions (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            timestamp TEXT,\n            applicant_id TEXT,\n            default_probability REAL,\n            risk_band TEXT,\n            top_shap_drivers TEXT\n        )\n    ')
    conn.execute('\n        CREATE TABLE IF NOT EXISTS audit_chat (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            timestamp TEXT,\n            question TEXT,\n            generated_sql TEXT,\n            answer TEXT,\n            error TEXT\n        )\n    ')
    conn.commit()
    conn.close()
    logger.info('Audit tables ready.')

def log_prediction(applicant_id, probability: float, band: str, top_drivers: dict=None) -> None:
    conn = _get_conn()
    conn.execute('INSERT INTO audit_predictions (timestamp, applicant_id, default_probability, risk_band, top_shap_drivers) VALUES (?, ?, ?, ?, ?)', (datetime.now(timezone.utc).isoformat(), str(applicant_id), float(probability), band, json.dumps(top_drivers) if top_drivers else None))
    conn.commit()
    conn.close()

def log_chat(question: str, sql: str=None, answer: str=None, error: str=None) -> None:
    conn = _get_conn()
    conn.execute('INSERT INTO audit_chat (timestamp, question, generated_sql, answer, error) VALUES (?, ?, ?, ?, ?)', (datetime.now(timezone.utc).isoformat(), question, sql, answer, error))
    conn.commit()
    conn.close()

def get_recent_predictions(limit: int=50):
    conn = _get_conn()
    init_audit_tables()
    rows = conn.execute('SELECT * FROM audit_predictions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return rows

def get_recent_chats(limit: int=50):
    conn = _get_conn()
    init_audit_tables()
    rows = conn.execute('SELECT * FROM audit_chat ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return rows
import re
import sqlite3
import pandas as pd
from src.utils.config import config
from src.utils.logger import get_logger
logger = get_logger(__name__)
FORBIDDEN_KEYWORDS = ('INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'ATTACH', 'PRAGMA')

def validate_sql(query: str) -> None:
    stripped = query.strip().rstrip(';')
    if not re.match('(?is)^\\s*SELECT\\b', stripped):
        raise ValueError('Only SELECT statements are permitted.')
    upper = stripped.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in upper:
            raise ValueError(f'Query contains forbidden keyword: {kw}')
    if ';' in stripped:
        raise ValueError('Multiple statements are not permitted.')

def run_query(query: str, db_path: str=None) -> pd.DataFrame:
    validate_sql(query)
    db_path = db_path or config.SQL_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        result = pd.read_sql_query(query, conn)
        logger.info(f'Query returned {len(result)} rows')
        return result
    finally:
        conn.close()

def build_db_from_dataframe(df: pd.DataFrame, db_path: str=None, table_name: str='applicants') -> None:
    db_path = db_path or config.SQL_DB_PATH
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"Loaded {len(df)} rows into '{table_name}' at {db_path}")
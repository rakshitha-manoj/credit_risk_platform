import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '') or os.getenv('GROQ_API_KEY', '')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
    LLM_MODEL = os.getenv('LLM_MODEL', 'openai/gpt-oss-120b')
    DATA_DIR = os.getenv('DATA_DIR', './data')
    MODELS_DIR = os.getenv('MODELS_DIR', './models')
    SQL_DB_PATH = os.getenv('SQL_DB_PATH', './sql/credit_risk.db')
    APP_PORT = int(os.getenv('APP_PORT', '8501'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    RISK_BAND_THRESHOLDS = {'low': 0.05, 'medium': 0.15}
    TARGET_COL = 'TARGET'
    ID_COL = 'SK_ID_CURR'
config = Config()
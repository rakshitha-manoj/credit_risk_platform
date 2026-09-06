import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger
from src.utils.config import config
logger = get_logger(__name__)
DAYS_SENTINEL = 365243

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.startswith('DAYS_'):
            df[col] = df[col].replace(DAYS_SENTINEL, pd.NA)
    nunique = df.nunique(dropna=False)
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        logger.info(f'Dropping {len(drop_cols)} constant columns')
        df = df.drop(columns=drop_cols)
    return df

def encode_and_impute(df: pd.DataFrame, id_col: str, target_col: str):
    df = df.copy()
    encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in (id_col, target_col)]
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    logger.info(f'Encoded {len(cat_cols)} categorical, imputed {len(num_cols)} numeric columns')
    return (df, encoders)

def preprocess(df: pd.DataFrame):
    df = clean(df)
    df, encoders = encode_and_impute(df, config.ID_COL, config.TARGET_COL)
    return (df, encoders)
import os
import numpy as np
import pandas as pd
from src.utils.config import config
from src.utils.helpers import load_artifact, risk_band
from src.data.preprocessor import clean

def load_pipeline():
    model = load_artifact(f'{config.MODELS_DIR}/model.joblib')
    encoders = load_artifact(f'{config.MODELS_DIR}/encoders.joblib')
    feature_cols = load_artifact(f'{config.MODELS_DIR}/feature_cols.joblib')
    return (model, encoders, feature_cols)

def preprocess_for_inference(df_raw: pd.DataFrame, encoders: dict=None, feature_cols: list=None, medians: dict=None) -> pd.DataFrame:
    if encoders is None or feature_cols is None:
        _, loaded_encoders, loaded_cols = load_pipeline()
        encoders = encoders if encoders is not None else loaded_encoders
        feature_cols = feature_cols if feature_cols is not None else loaded_cols
    if medians is None:
        medians_path = f'{config.MODELS_DIR}/medians.joblib'
        medians = load_artifact(medians_path) if os.path.exists(medians_path) else {}
    df = clean(df_raw)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype(str)
            df[col] = df[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0)
        elif df[col].isna().any():
            fill_val = medians.get(col, df[col].median() if df[col].dtype != 'object' else 0)
            df[col] = df[col].fillna(fill_val)
    return df[feature_cols]

def predict(df_raw: pd.DataFrame, model=None, encoders=None, feature_cols=None, calibrator=None, medians=None) -> pd.DataFrame:
    if model is None or encoders is None or feature_cols is None:
        m, enc, cols = load_pipeline()
        model = model or m
        encoders = encoders or enc
        feature_cols = feature_cols or cols
    X = preprocess_for_inference(df_raw, encoders=encoders, feature_cols=feature_cols, medians=medians)
    probs = model.predict_proba(X)[:, 1]
    if calibrator is None:
        calibrator_path = f'{config.MODELS_DIR}/calibrator.joblib'
        if os.path.exists(calibrator_path):
            calibrator = load_artifact(calibrator_path)
    if calibrator is not None:
        p_clip = np.clip(probs, 1e-07, 1.0 - 1e-07)
        logit = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        probs = calibrator.predict_proba(logit)[:, 1]
    ids = df_raw[config.ID_COL] if config.ID_COL in df_raw.columns else range(len(df_raw))
    bands = [risk_band(p, config.RISK_BAND_THRESHOLDS) for p in probs]
    try:
        from src.utils.audit import init_audit_tables, log_prediction
        init_audit_tables()
        for applicant_id, prob, band in zip(ids, probs, bands):
            log_prediction(applicant_id, prob, band)
    except Exception:
        pass
    return pd.DataFrame({config.ID_COL: ids, 'default_probability': probs, 'risk_band': bands})
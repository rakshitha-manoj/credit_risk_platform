import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data.loader import build_full_dataset
from src.data.preprocessor import preprocess
from src.utils.config import config
from src.utils.helpers import save_artifact
from src.utils.logger import get_logger
logger = get_logger(__name__)

def get_feature_columns(df):
    return [c for c in df.columns if c not in (config.ID_COL, config.TARGET_COL)]

def train_model():
    raw = build_full_dataset(config.DATA_DIR)
    df, encoders = preprocess(raw)
    feature_cols = get_feature_columns(df)
    X, y = (df[feature_cols], df[config.TARGET_COL])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f'Class imbalance ratio (neg/pos): {pos_weight:.2f}')
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, scale_pos_weight=pos_weight, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(50)])
    save_artifact(model, f'{config.MODELS_DIR}/model.joblib')
    save_artifact(encoders, f'{config.MODELS_DIR}/encoders.joblib')
    save_artifact(feature_cols, f'{config.MODELS_DIR}/feature_cols.joblib')
    val_probs = model.predict_proba(X_val)[:, 1]
    p_clipped = np.clip(val_probs, 1e-07, 1.0 - 1e-07)
    val_logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(solver='lbfgs')
    calibrator.fit(val_logits, y_val)
    save_artifact(calibrator, f'{config.MODELS_DIR}/calibrator.joblib')
    medians = df[feature_cols].select_dtypes(include=[np.number]).median().to_dict()
    save_artifact(medians, f'{config.MODELS_DIR}/medians.joblib')
    logger.info('Model and artifacts saved successfully.')
    return (model, X_val, y_val, feature_cols)
if __name__ == '__main__':
    train_model()
import os
import joblib

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_artifact(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)

def load_artifact(path: str):
    return joblib.load(path)

def risk_band(prob: float, thresholds: dict) -> str:
    if prob < thresholds['low']:
        return 'Low'
    elif prob < thresholds['medium']:
        return 'Medium'
    return 'High'
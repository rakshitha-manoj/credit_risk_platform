import shap
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from src.utils.config import config
from src.utils.helpers import load_artifact
from src.utils.logger import get_logger
logger = get_logger(__name__)

def get_explainer():
    model = load_artifact(f'{config.MODELS_DIR}/model.joblib')
    return (shap.TreeExplainer(model), model)

def explain_global(X_sample: pd.DataFrame):
    explainer, _ = get_explainer()
    shap_values = explainer.shap_values(X_sample)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    importance = pd.Series(abs(vals).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)
    return importance

def explain_instance(X_row: pd.DataFrame) -> pd.Series:
    explainer, _ = get_explainer()
    shap_values = explainer.shap_values(X_row)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    contributions = pd.Series(vals[0], index=X_row.columns).sort_values(key=abs, ascending=False)
    return contributions

def derive_rules_descriptive(X_sample: pd.DataFrame, top_n: int=5) -> list:
    importance = explain_global(X_sample)
    rules = []
    for feature in importance.head(top_n).index:
        median = X_sample[feature].median()
        rules.append(f'Applicants with {feature} deviating from the typical value (median: {median:.2f}) are a significant driver of risk classification.')
    logger.info(f'Derived {len(rules)} descriptive rules from top SHAP features')
    return rules

def derive_rules_surrogate(X_sample: pd.DataFrame, max_depth: int=4) -> dict:
    _, model = get_explainer()
    surrogate_labels = model.predict(X_sample)
    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    surrogate.fit(X_sample, surrogate_labels)
    fidelity = surrogate.score(X_sample, surrogate_labels)
    rule_text = export_text(surrogate, feature_names=list(X_sample.columns))
    logger.info(f'Surrogate tree fidelity to base model: {fidelity:.4f}')
    return {'rules_text': rule_text, 'fidelity': fidelity}

def derive_rules(X_sample: pd.DataFrame, top_n: int=5, max_depth: int=4) -> dict:
    return {'descriptive': derive_rules_descriptive(X_sample, top_n=top_n), 'surrogate': derive_rules_surrogate(X_sample, max_depth=max_depth)}
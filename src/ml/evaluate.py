from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, classification_report
from src.utils.logger import get_logger
logger = get_logger(__name__)

def evaluate_model(model, X_val, y_val) -> dict:
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {'roc_auc': roc_auc_score(y_val, probs), 'pr_auc': average_precision_score(y_val, probs), 'brier_score': brier_score_loss(y_val, probs)}
    logger.info(f'ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Brier: {metrics['brier_score']:.4f}')
    logger.info('\n' + classification_report(y_val, preds))
    return metrics
if __name__ == '__main__':
    from src.ml.train import train_model
    model, X_val, y_val, _ = train_model()
    evaluate_model(model, X_val, y_val)
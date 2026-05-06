from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
) 


def get_probability_scores(estimator, X_test):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X_test)

    return None


def evaluate_model(model_name, year, estimator, X_test, y_test):
    y_test = y_test.squeeze()

    y_pred = estimator.predict(X_test)
    y_score = get_probability_scores(estimator, X_test)

    if y_score is not None:
        roc_auc = roc_auc_score(y_test, y_score)
    else:
        roc_auc = None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "year": year,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def evaluate_thresholds(model_name, year, estimator, X_test, y_test):
    y_test = y_test.squeeze()

    # Threshold testing only makes sense for models with predict_proba
    if not hasattr(estimator, "predict_proba"):
        return []

    y_score = estimator.predict_proba(X_test)[:, 1]

    rows = []

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


        rows.append({
            "year": year,
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return rows

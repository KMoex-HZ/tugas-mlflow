import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.data
import mlflow.sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from data import X_test, y_test, test_df
from analysis import get_best_runs

best = get_best_runs().iloc[0]
best_run_id = best["run_id"]
best_model_uri = best["tags.model_uri"]
print("Evaluasi:", best["tags.mlflow.runName"], best_run_id)

model = mlflow.sklearn.load_model(best_model_uri)
y_pred = model.predict(X_test)
score = (
    model.predict_proba(X_test)[:, 1]
    if hasattr(model, "predict_proba")
    else model.decision_function(X_test)
)

test_metrics = {
    "test_recall":    recall_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_f1":        f1_score(y_test, y_pred),
    "test_roc_auc":   roc_auc_score(y_test, score),
}

with mlflow.start_run(run_id=best_run_id):  # tambahkan ke run yang SAMA
    mlflow.log_metrics(test_metrics)
    mlflow.log_input(
        mlflow.data.from_pandas(test_df, source="data/test.csv",
                                name="bc-test", targets="target"),
        context="testing",
    )
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["benign", "malignant"]
    )
    mlflow.log_figure(disp.figure_, "plots/confusion_matrix_test.png")
    plt.close(disp.figure_)
    mlflow.set_tag("test_evaluated", "true")

print({k: round(v, 3) for k, v in test_metrics.items()})
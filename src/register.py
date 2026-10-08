import mlflow
from mlflow import MlflowClient

from data import data_md5
from analysis import get_best_runs

client, NAME = MlflowClient(), "bc-malignant-classifier"
MIN_PRECISION, MARGIN = 0.90, 0.005

best = get_best_runs().iloc[0]
best_run_id = best["run_id"]
best_model_uri = best["tags.model_uri"]
test_metrics = client.get_run(best_run_id).data.metrics  # hasil L6

mv = mlflow.register_model(best_model_uri, NAME)  # -> versi baru
client.update_model_version(
    NAME, mv.version,
    description=f"Dipilih otomatis: max cv_recall dgn precision>=0.90. data_md5={data_md5}",
)
client.set_model_version_tag(NAME, mv.version, "data_md5", data_md5)
client.set_registered_model_alias(NAME, "challenger", mv.version)


def champion_recall():
    try:
        champ = client.get_model_version_by_alias(NAME, "champion")
        return client.get_run(champ.run_id).data.metrics["test_recall"]
    except Exception:  # belum ada champion
        return -1.0


passed = (
    test_metrics["test_precision"] >= MIN_PRECISION
    and test_metrics["test_recall"] >= champion_recall() + MARGIN
)

client.set_model_version_tag(
    NAME, mv.version, "validation_status", "approved" if passed else "rejected"
)
if passed:
    client.set_registered_model_alias(NAME, "champion", mv.version)  # promosi

print("Versi:", mv.version, "| lolos quality gate:", passed)
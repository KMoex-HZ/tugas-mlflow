import mlflow

EXPERIMENT = "bc-malignant-detection"


def get_best_runs():
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=(
            "tags.candidate = 'true' "                  # hanya run yang punya model
            "and metrics.cv_precision_mean >= 0.90"     # satisficing metric
        ),
        order_by=[
            "metrics.cv_recall_mean DESC",              # optimizing metric
            "metrics.cv_f1_mean DESC",                  # tie-breaker
        ],
    )


if __name__ == "__main__":
    runs = get_best_runs()
    cols = [
        "tags.mlflow.runName", "tags.model_family",
        "metrics.cv_recall_mean", "metrics.cv_recall_std",
        "metrics.cv_precision_mean",
    ]
    print(runs[cols].head(5).to_string(index=False))
    print("Kandidat terpilih:", runs.iloc[0]["tags.mlflow.runName"], runs.iloc[0]["run_id"])
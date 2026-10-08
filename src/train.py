import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import X_train, y_train, train_ds, COMMON_TAGS
from tracking import run_experiment, CV, SCORING

# ---------- L2: Baseline (H1, H2) ----------
run_experiment(
    "00-dummy-most-frequent",
    Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
    tags={"stage": "baseline", "model_family": "dummy"},
    note="Batas bawah: selalu menebak kelas mayoritas (benign).",
)

run_experiment(
    "01-logreg-noscale",
    Pipeline([("clf", LogisticRegression(max_iter=5000))]),
    tags={"stage": "baseline", "model_family": "logreg"},
    note="H1: model linear tanpa preprocessing.",
)

run_experiment(
    "02-logreg-scaled",
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
    tags={"stage": "baseline", "model_family": "logreg"},
    note="H2: hanya menambah StandardScaler, variabel lain tetap.",
)

# ---------- L3: Model selection ----------
candidates = {
    "knn":  KNeighborsClassifier(n_neighbors=5),
    "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "rf":   RandomForestClassifier(n_estimators=200, random_state=42),
    "gb":   GradientBoostingClassifier(random_state=42),
    "svc":  SVC(kernel="rbf", random_state=42),
}

for i, (family, clf) in enumerate(candidates.items(), start=3):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    run_experiment(
        f"{i:02d}-{family}-default",
        pipe,
        tags={"stage": "model-selection", "model_family": family},
        note="Hyperparameter default; preprocessing identik untuk semua.",
    )

# ---------- L4: Tuning SVC dengan nested runs (H3) ----------
pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])
grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}  # 12 kombinasi
gs = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, refit="recall", n_jobs=-1)

with mlflow.start_run(
    run_name="10-svc-gridsearch",
    tags={**COMMON_TAGS, "stage": "tuning", "model_family": "svc", "candidate": "true"},
):
    mlflow.log_input(train_ds, context="training")
    mlflow.log_dict(grid, "search_space.json")  # ruang pencarian = artefak
    gs.fit(X_train, y_train)
    res = gs.cv_results_

    for i, params in enumerate(res["params"]):  # 1 trial = 1 child run
        with mlflow.start_run(run_name=f"svc-trial-{i:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({f"cv_{m}_mean": res[f"mean_test_{m}"][i] for m in SCORING})
            mlflow.log_metrics({f"cv_{m}_std": res[f"std_test_{m}"][i] for m in SCORING})

    b = gs.best_index_

    # parent = ringkasan terbaik
    mlflow.log_params(gs.best_params_)
    mlflow.log_metrics({f"cv_{m}_mean": res[f"mean_test_{m}"][b] for m in SCORING})
    mlflow.log_metrics({f"cv_{m}_std": res[f"std_test_{m}"][b] for m in SCORING})
    info = mlflow.sklearn.log_model(
        gs.best_estimator_,
        name="model",
        signature=infer_signature(X_train, gs.best_estimator_.predict(X_train)),
        input_example=X_train.head(3),
    )
    mlflow.set_tag("model_uri", info.model_uri)
    print("Best params:", gs.best_params_)
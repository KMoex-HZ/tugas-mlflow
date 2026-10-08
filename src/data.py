import hashlib
import os

import mlflow
import mlflow.data
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

EXPERIMENT = "bc-malignant-detection"
mlflow.set_experiment(EXPERIMENT)

os.makedirs("data", exist_ok=True)

df = load_breast_cancer(as_frame=True).frame
df["target"] = 1 - df["target"]  # 1 = malignant (kelas positif)

# Sidik jari data: berubah satu nilai pun -> hash berbeda
data_md5 = hashlib.md5(
    pd.util.hash_pandas_object(df, index=True).values
).hexdigest()

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["target"], random_state=42
)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)  # test set: dikunci sampai L6

X_train, y_train = train_df.drop(columns="target"), train_df["target"]
X_test, y_test = test_df.drop(columns="target"), test_df["target"]

# Objek Dataset MLflow -> tampil di tab "Datasets" pada setiap run
train_ds = mlflow.data.from_pandas(
    train_df, source="data/train.csv", name="bc-train", targets="target"
)

COMMON_TAGS = {
    "data_md5": data_md5,
    "split_seed": "42",
    "dataset": "sklearn-breast-cancer",
}

if __name__ == "__main__":
    print("Jumlah sampel :", len(df))
    print("Train / Test  :", len(train_df), "/", len(test_df))
    print("data_md5      :", data_md5)
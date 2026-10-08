import requests
import mlflow

from data import X_test

# Lewat REST API
payload = {"dataframe_split": X_test.head(3).to_dict(orient="split")}
r = requests.post("http://127.0.0.1:5001/invocations", json=payload)
print("REST   :", r.json())  # 1 = malignant

# Langsung di Python (batch scoring)
model = mlflow.pyfunc.load_model("models:/bc-malignant-classifier@champion")
print("PyFunc :", model.predict(X_test.head(3)))
# Tugas MLflow - Hands-on 2: Deteksi Tumor Ganas (Breast Cancer Wisconsin)

**Nama:** Khairunnisa Maharani
**NIM:** 123450071
**Kelas:** RB

Studi kasus end-to-end MLflow + scikit-learn: dari hipotesis, eksperimen,
tuning, model registry, serving, hingga reproduksi hasil.

## Skenario
- Dataset: `sklearn.datasets.load_breast_cancer` (569 sampel, 30 fitur)
- Kelas positif: 1 = malignant
- Optimizing metric: Recall
- Satisficing metric: Precision >= 0.90
- Evaluasi: train 80% (5-fold Stratified CV), test 20% (sekali di akhir)

## Struktur
```
src/data.py       # L1: load, split, hash data
src/tracking.py   # fungsi logging standar run_experiment()
src/train.py      # L2-L4: baseline, model selection, tuning
src/analysis.py   # L5: seleksi kandidat via search_runs
src/evaluate.py   # L6: evaluasi test set
src/register.py   # L7: registry + quality gate
src/predict.py    # L8: serving
src/reproduce.py  # L9: lineage & reproduce
```

## Cara menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000

# Terminal 2
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/data.py
python src/train.py
python src/analysis.py
python src/evaluate.py
python src/register.py

# Terminal 3 (serving)
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/bc-malignant-classifier@champion" -p 5001 --env-manager local

# Terminal 2
python src/predict.py
python src/reproduce.py
```

## Hasil ringkas
| Metrik | Nilai |
|---|---|
| Model terbaik | SVC (C=10, gamma=scale), run `10-svc-gridsearch` |
| CV recall (mean ± std) | 0.965 ± 0.029 |
| CV precision | 0.971 |
| Test recall | 0.929 |
| Test precision | 1.000 |
| Test F1 | 0.963 |
| Test ROC-AUC | 0.993 |
| Quality gate | Lolos, alias `@champion` (versi 1) |

Screenshot MLflow UI ada di folder `screenshots/`.

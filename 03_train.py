from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "training" / "training_samples.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "elevation_m",
    "slope_degrees",
    "roughness_5px",
    "NDVI",
    "NDMI",
    "BSI",
]

df = pd.read_csv(CSV_PATH)
missing = [c for c in FEATURES + ["site_id", "moon_label", "mars_label"] if c not in df.columns]
if missing:
    raise ValueError(f"Eksik sütunlar: {missing}")

X = df[FEATURES]
groups = df["site_id"]

for target_col, model_name in [("moon_label", "moon_model.joblib"), ("mars_label", "mars_model.joblib")]:
    print(f"\n==========================================")
    print(f"MODEL EĞİTİLİYOR: {target_col.upper()}")
    print(f"==========================================")
    y = df[target_col]

    sgkf = StratifiedGroupKFold(n_splits=4)
    oof_preds = np.zeros(len(df), dtype=int)
    oof_probs = np.zeros(len(df), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = clf.predict(X.iloc[test_idx])
        oof_probs[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

    print(classification_report(y, oof_preds, digits=3))
    print(f"4-Katmanlı Çapraz Doğrulama ROC AUC: {roc_auc_score(y, oof_probs):.4f}")

    # Tüm 12 saha ile nihai modeli eğit ve kaydet
    final_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X, y)
    out_path = MODELS_DIR / model_name
    joblib.dump(final_clf, out_path)
    print(f"Nihai model kaydedildi: {out_path}")
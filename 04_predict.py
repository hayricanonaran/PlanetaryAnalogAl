from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
PRED_DIR = OUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "elevation_m",
    "slope_degrees",
    "roughness_5px",
    "NDVI",
    "NDMI",
    "BSI",
]

moon_model = joblib.load(MODELS_DIR / "moon_model.joblib")
mars_model = joblib.load(MODELS_DIR / "mars_model.joblib")

feature_files = sorted(OUT_DIR.glob("*_features.tif"))
if not feature_files:
    raise FileNotFoundError("Özellik haritası bulunamadı! Önce 02_features.py çalıştırılmalı.")

for fpath in feature_files:
    site_name = fpath.stem.replace("_features", "")
    print(f"Tahmin haritası üretiliyor: {site_name.upper()}...")

    with rasterio.open(fpath) as src:
        data = src.read()  # (6, H, W)
        profile = src.profile.copy()

    _, h, w = data.shape
    flat = data.reshape(6, -1).T
    df_pixels = pd.DataFrame(flat, columns=FEATURES)

    # Geçersiz/NaN değerleri temizle
    valid_mask = np.isfinite(flat).all(axis=1)
    moon_probs = np.full(h * w, np.nan, dtype=np.float32)
    mars_probs = np.full(h * w, np.nan, dtype=np.float32)

    if valid_mask.any():
        X_valid = df_pixels.loc[valid_mask]
        moon_probs[valid_mask] = moon_model.predict_proba(X_valid)[:, 1]
        mars_probs[valid_mask] = mars_model.predict_proba(X_valid)[:, 1]

    # 2 bantlı çıktı (1. Bant: Ay Analog Skoru, 2. Bant: Mars Analog Skoru)
    out_stack = np.stack([
        moon_probs.reshape(h, w),
        mars_probs.reshape(h, w),
    ])

    out_profile = profile.copy()
    out_profile.update(
        count=2,
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
    )

    out_file = PRED_DIR / f"{site_name}_analog_scores.tif"
    with rasterio.open(out_file, "w", **out_profile) as dst:
        dst.write(out_stack)
        dst.set_band_description(1, "Moon_Analog_Probability")
        dst.set_band_description(2, "Mars_Analog_Probability")

print(f"\nTAMAMLANDI! Tüm tahmin haritaları kaydedildi: {PRED_DIR}")
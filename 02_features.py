from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
TRAIN_DIR = ROOT / "data" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# 12 Sahanın Ay/Mars Etiketleri (moon_label, mars_label)
SITE_LABELS = {
    # Mars Analogları
    "salda": (0, 1),
    "atacama": (0, 1),
    "riotinto": (0, 1),
    "mojave": (0, 1),
    # Ay Analogları
    "askja": (1, 0),
    "cinderlake": (1, 0),
    "etna": (1, 0),
    "lanzarote": (1, 0),
    # Negatif Kontrol Sahaları
    "control_forest": (0, 0),
    "control_agri": (0, 0),
    "control_urban": (0, 0),
    "control_alpine": (0, 0),

    # YENI KESIF SAHALARI
    "levent_vadisi": (-1, -1),
    "karanlik_kanyon": (-1, -1),
    "tuz_golu": (-1, -1),
    "kula_volkanik": (-1, -1),
    "meke_krateri": (-1, -1),
}

all_samples = []

for site_name, (moon_lbl, mars_lbl) in SITE_LABELS.items():
    if moon_lbl == -1: pass
    site_dir = PROCESSED_DIR / site_name
    if not (site_dir / "DEM.tif").exists():
        print(f"UYARI: {site_name} verisi eksik, atlanıyor.")
        continue

    print(f"Özellikler hesaplanıyor: {site_name.upper()}...")
    with rasterio.open(site_dir / "B02_blue.tif") as src:
        blue = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(site_dir / "B04_red.tif") as src:
        red = src.read(1).astype(np.float32)
    with rasterio.open(site_dir / "B08_nir.tif") as src:
        nir = src.read(1).astype(np.float32)
    with rasterio.open(site_dir / "B11_swir.tif") as src:
        swir = src.read(1).astype(np.float32)
    with rasterio.open(site_dir / "DEM.tif") as src:
        dem = src.read(1).astype(np.float32)

    # 1. Topografik özellikler: Yükseklik, Eğim ve Pürüzlülük
    dy, dx = np.gradient(dem, 10.0)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    roughness = np.sqrt(np.maximum(0, uniform_filter(dem**2, 5) - uniform_filter(dem, 5)**2))

    # 2. Spektral indeksler
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndmi = (nir - swir) / (nir + swir + eps)
    bsi = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)

    # Su tespiti için McFeeters NDWI (Mavi ve NIR farkı; su yüzeyleri > 0.05 olur)
    water_index = (blue - nir) / (blue + nir + eps)

    # 6 katmanlı özellik haritasını kaydet
    stack = np.stack([dem, slope, roughness, ndvi, ndmi, bsi]).astype(np.float32)
    out_profile = profile.copy()
    out_profile.update(count=6, dtype="float32", compress="lzw")
    with rasterio.open(OUT_DIR / f"{site_name}_features.tif", "w", **out_profile) as dst:
        dst.write(stack)

    # Su yüzeylerini (water_index > 0.05) ve geçersiz pikselleri eğitimden hariç tut
    valid_mask = (
        np.isfinite(dem)
        & (dem > -500)
        & (blue > 0)
        & (water_index <= 0.05)
    )
    rows, cols = np.where(valid_mask)
    if len(rows) > 600:
        idx = np.linspace(0, len(rows) - 1, 600, dtype=int)
        rows, cols = rows[idx], cols[idx]

    df_site = pd.DataFrame({
        "site_id": site_name,
        "elevation_m": dem[rows, cols],
        "slope_degrees": slope[rows, cols],
        "roughness_5px": roughness[rows, cols],
        "NDVI": ndvi[rows, cols],
        "NDMI": ndmi[rows, cols],
        "BSI": bsi[rows, cols],
        "moon_label": moon_lbl,
        "mars_label": mars_lbl,
    })
    all_samples.append(df_site)

train_df = pd.concat(all_samples, ignore_index=True)
csv_path = TRAIN_DIR / "training_samples.csv"
train_df.to_csv(csv_path, index=False)

print(f"\nTAMAMLANDI!")
print(f"12 özellik haritası: {OUT_DIR}")
print(f"Su maskeli temiz eğitim tablosu ({len(train_df)} örnek): {csv_path}")

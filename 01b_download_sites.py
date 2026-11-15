import os
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# 12 Farklı Bilimsel Eğitim ve Kontrol Sahası (min_lon, min_lat, max_lon, max_lat)
SITES = {
    # MARS ANALOGLARI
    "salda": {"bbox": [29.62, 37.50, 29.74, 37.60], "date": "2024-06-01/2024-09-30"},
    "atacama": {"bbox": [-69.95, -24.15, -69.83, -24.05], "date": "2024-01-01/2024-12-31"},
    "riotinto": {"bbox": [-6.62, 37.66, -6.50, 37.76], "date": "2024-06-01/2024-09-30"},
    "mojave": {"bbox": [-116.85, 36.45, -116.73, 36.55], "date": "2024-05-01/2024-09-30"},

    # AY ANALOGLARI
    "askja": {"bbox": [-16.82, 65.00, -16.66, 65.08], "date": "2024-07-01/2024-08-31"},
    "cinderlake": {"bbox": [-111.55, 35.30, -111.43, 35.40], "date": "2024-05-01/2024-09-30"},
    "etna": {"bbox": [14.95, 37.70, 15.05, 37.78], "date": "2024-06-01/2024-09-30"},
    "lanzarote": {"bbox": [-13.80, 28.97, -13.68, 29.05], "date": "2024-05-01/2024-09-30"},

    # NEGATİF KONTROL SAHALARI (Dünya Orman, Tarım, Şehir, Dağ)
    "control_forest": {"bbox": [31.55, 40.65, 31.67, 40.75], "date": "2024-06-01/2024-09-30"},
    "control_agri": {"bbox": [32.70, 37.80, 32.82, 37.90], "date": "2024-06-01/2024-09-30"},
    "control_urban": {"bbox": [28.80, 41.00, 28.92, 41.10], "date": "2024-06-01/2024-09-30"},
    "control_alpine": {"bbox": [41.10, 40.80, 41.22, 40.90], "date": "2024-07-15/2024-09-15"},
    # YENI KESIF SAHALARI (Turkiye)
    "levent_vadisi": {"bbox": [37.86, 38.35, 37.98, 38.45], "date": "2024-06-01/2024-09-30"},
    "karanlik_kanyon": {"bbox": [38.43, 39.22, 38.55, 39.33], "date": "2024-06-01/2024-09-30"},
    "tuz_golu": {"bbox": [33.34, 38.70, 33.46, 38.80], "date": "2024-07-01/2024-09-30"},
    "kula_volkanik": {"bbox": [28.46, 38.53, 28.58, 38.63], "date": "2024-06-01/2024-09-30"},
    "meke_krateri": {"bbox": [33.58, 37.63, 33.70, 37.73], "date": "2024-06-01/2024-09-30"},
}

band_map = {
    "B02_blue.tif": "blue",
    "B04_red.tif": "red",
    "B08_nir.tif": "nir",
    "B11_swir.tif": "swir16",
}

catalog = Client.open("https://earth-search.aws.element84.com/v1")

for site_name, info in SITES.items():
    out_dir = PROCESSED_DIR / site_name
    if (out_dir / "DEM.tif").exists():
        print(f"[ATLANDI] {site_name.upper()} zaten indirilmiş.")
        continue

    print(f"\n==========================================")
    print(f"SAHA İŞLENİYOR: {site_name.upper()}")
    print(f"==========================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = info["bbox"]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=info["date"],
        query={"eo:cloud_cover": {"lt": 12}},
    )
    items = list(search.items())
    if not items:
        print(f"UYARI: {site_name} için bulutsuz görüntü bulunamadı, atlanıyor.")
        continue

    item = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
    print(f"Seçilen görüntü: {item.id} (Bulut: %{item.properties.get('eo:cloud_cover')})")

    with rasterio.open(item.assets["blue"].href) as ref_src:
        dst_crs = ref_src.crs
        dst_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox)
        res = 10.0
        width = int((dst_bounds[2] - dst_bounds[0]) / res)
        height = int((dst_bounds[3] - dst_bounds[1]) / res)
        dst_transform = rasterio.transform.from_bounds(*dst_bounds, width, height)

        profile = ref_src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            width=width,
            height=height,
            dtype="float32",
            count=1,
            compress="lzw",
        )

    for out_name, asset_key in band_map.items():
        print(f"   -> {out_name} indiriliyor...")
        dst_array = np.zeros((height, width), dtype=np.float32)
        with rasterio.open(item.assets[asset_key].href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        with rasterio.open(out_dir / out_name, "w", **profile) as dst:
            dst.write(dst_array, 1)

    print("   -> DEM.tif indiriliyor...")
    dem_search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    dem_items = list(dem_search.items())
    if dem_items:
        dem_array = np.zeros((height, width), dtype=np.float32)
        with rasterio.open(dem_items[0].assets["data"].href) as dem_src:
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_array,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        with rasterio.open(out_dir / "DEM.tif", "w", **profile) as dst:
            dst.write(dem_array, 1)

print("\n12 SAHANIN TAMAMI HAZIR!")
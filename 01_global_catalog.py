
import csv
import time
from pathlib import Path
from pystac_client import Client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "sentinel2_global.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://stac.dataspace.copernicus.eu/v1/"
COLLECTION = "sentinel-2-l2a"

# İlk envanter için yakın tarih aralığı.
DATE_RANGE = "2024-01-01/2026-11-14"

# 10 derece hücreler. Her hücrede en fazla 20 kayıt.
STEP = 10
MAX_ITEMS_PER_CELL = 20
MAX_CLOUD = 20

client = Client.open(URL)

fields = [
    "item_id", "datetime", "cloud_cover",
    "west", "south", "east", "north"
]

rows = []
errors = []

print("Küresel Sentinel-2 katalog taraması başladı.")

for south in range(-60, 85, STEP):
    north = min(south + STEP, 85)

    for west in range(-180, 180, STEP):
        east = west + STEP
        bbox = [west, south, east, north]

        try:
            search = client.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime=DATE_RANGE,
                query={
                    "eo:cloud_cover": {
                        "lt": MAX_CLOUD
                    }
                },
                max_items=MAX_ITEMS_PER_CELL
            )

            count = 0

            for item in search.items():
                props = item.properties
                item_bbox = item.bbox or bbox

                rows.append({
                    "item_id": item.id,
                    "datetime": props.get("datetime"),
                    "cloud_cover": props.get("eo:cloud_cover"),
                    "west": item_bbox[0],
                    "south": item_bbox[1],
                    "east": item_bbox[2],
                    "north": item_bbox[3],
                })

                count += 1

            print(
                f"Tamamlandı: {west},{south} "
                f"- kayıt: {count}"
            )

        except Exception as exc:
            errors.append({
                "bbox": bbox,
                "error": str(exc)
            })
            print("Hücre hatası:", bbox, exc)

        # Servise aşırı yük bindirmemek için.
        time.sleep(0.2)

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

error_file = OUT.with_name("catalog_errors.txt")
with error_file.open("w", encoding="utf-8") as f:
    for error in errors:
        f.write(str(error) + "\n")

print("--------------------------------")
print("Katalog taraması tamamlandı.")
print("Bulunan kayıt:", len(rows))
print("Hatalı hücre:", len(errors))
print("CSV:", OUT)
print("Hatalar:", error_file)

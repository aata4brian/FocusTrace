"""Prepare downloaded C1 photos for the simulated social feed.

Put 12 source images in content/C1/downloads, then run:
    python scripts/prepare_c1_photos.py

The script center-crops each image to 4:5, resizes to 1080x1350,
exports JPEGs 1.jpg..12.jpg, and updates experiment.json paths.
Original downloads are never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
SOURCE = CONTENT / "C1" / "downloads"
OUTPUT = CONTENT / "C1" / "media"
MANIFEST = CONTENT / "experiment.json"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
TARGET = (1080, 1350)  # 4:5 portrait, Instagram-style


def natural_key(path: Path) -> tuple:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def crop_4_5(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    target_ratio = 4 / 5
    current = w / h
    if current > target_ratio:
        new_w = round(h * target_ratio)
        left = (w - new_w) // 2
        image = image.crop((left, 0, left + new_w, h))
    elif current < target_ratio:
        new_h = round(w / target_ratio)
        top = (h - new_h) // 2
        image = image.crop((0, top, w, top + new_h))
    return image.resize(TARGET, Image.Resampling.LANCZOS)


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED),
        key=natural_key,
    )
    if len(files) != 12:
        raise SystemExit(
            f"Butuh tepat 12 foto di {SOURCE}. Ditemukan {len(files)}. "
            "Nama yang disarankan: 1.jpg ... 12.jpg."
        )

    for index, source in enumerate(files, start=1):
        with Image.open(source) as image:
            prepared = crop_4_5(image)
            prepared.save(
                OUTPUT / f"{index}.jpg",
                "JPEG",
                quality=90,
                optimize=True,
                progressive=True,
            )
        print(f"[{index:02d}/12] {source.name} -> C1/media/{index}.jpg")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feed = data.get("C1", {}).get("feed", [])
    if len(feed) != 12:
        raise SystemExit("experiment.json harus memiliki tepat 12 post C1 untuk script ini.")
    for index, post in enumerate(feed, start=1):
        post["image"] = f"C1/media/{index}.jpg"
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Selesai. experiment.json juga sudah diarahkan ke 1.jpg ... 12.jpg.")


if __name__ == "__main__":
    main()

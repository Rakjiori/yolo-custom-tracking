from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

SRC = Path(r"C:\yolo_gun\photos_raw")
DST = Path(r"C:\yolo_gun\photos_jpg")
MAX_SIDE = 2048


def convert(p: Path) -> str:
    register_heif_opener()
    out = DST / (p.stem + ".jpg")
    if out.exists():
        return "skip"
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(out, "JPEG", quality=92)
        return "ok"
    except Exception as e:
        return f"fail {p.name}: {e}"


if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"})
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(convert, files, chunksize=8))
    print("ok:", res.count("ok"), "| skip:", res.count("skip"), "| fail:", [r for r in res if r.startswith("fail")])

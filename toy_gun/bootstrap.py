import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"C:\yolo_gun")
IMG, LBL = ROOT / "photos_jpg", ROOT / "labels_auto"
BOOT = ROOT / "boot_ds"
CONF = 0.35

if __name__ == "__main__":
    shutil.rmtree(BOOT, ignore_errors=True)
    labeled = sorted(LBL.glob("*.txt"))
    random.seed(0); random.shuffle(labeled)
    n_val = len(labeled) // 10
    for split, items in (("val", labeled[:n_val]), ("train", labeled[n_val:])):
        (BOOT / split / "images").mkdir(parents=True); (BOOT / split / "labels").mkdir(parents=True)
        for t in items:
            shutil.copy2(IMG / f"{t.stem}.jpg", BOOT / split / "images")
            shutil.copy2(t, BOOT / split / "labels")
    (BOOT / "data.yaml").write_text(f"path: {BOOT.as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['toy gun']\n")

    model = YOLO(r"C:\yolo_project\yolo11s.pt")
    model.train(data=str(BOOT / "data.yaml"), epochs=40, imgsz=640, batch=32, device=0, workers=4,
                project=str(ROOT / "runs"), name="boot", exist_ok=True, verbose=False, plots=False)

    boot = YOLO(str(ROOT / "runs" / "boot" / "weights" / "best.pt"))
    missing = (ROOT / "missing.txt").read_text().split()
    still, added = [], 0
    for k in range(0, len(missing), 16):
        names = missing[k:k + 16]
        for n, r in zip(names, boot.predict([str(IMG / f"{n}.jpg") for n in names], conf=0.05, verbose=False)):
            b = r.boxes
            if len(b) == 0 or float(b.conf.max()) < CONF:
                still.append(n); continue
            i = int(b.conf.argmax()); x, y, w, h = b.xywhn[i].tolist()
            (LBL / f"{n}.txt").write_text(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"); added += 1
    (ROOT / "missing2.txt").write_text("\n".join(still))
    print(f"\n추가 라벨 {added}장 / 여전히 못 찾음 {len(still)}장 / 전체 라벨 {len(list(LBL.glob('*.txt')))}장")

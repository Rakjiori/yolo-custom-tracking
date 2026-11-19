from pathlib import Path

from ultralytics import YOLOWorld

ROOT = Path(r"C:\yolo_gun")
IMG = ROOT / "photos_jpg"
LBL = ROOT / "labels_auto"
PROMPTS = ["toy gun", "pistol", "revolver", "handgun"]
CONF = 0.15

if __name__ == "__main__":
    LBL.mkdir(exist_ok=True)
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(PROMPTS)
    files = sorted(IMG.glob("*.jpg"))
    missing, confs = [], []
    def batches(n=16):
        for k in range(0, len(files), n):
            yield from model.predict([str(p) for p in files[k:k + n]], conf=0.05, imgsz=960, verbose=False)

    for r in batches():
        name = Path(r.path).stem
        b = r.boxes
        if len(b) == 0 or float(b.conf.max()) < CONF:
            missing.append(name)
            (LBL / f"{name}.txt").unlink(missing_ok=True)
            continue
        i = int(b.conf.argmax())
        x, y, w, h = b.xywhn[i].tolist()
        (LBL / f"{name}.txt").write_text(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        confs.append(float(b.conf[i]))
    (ROOT / "missing.txt").write_text("\n".join(missing))
    confs.sort()
    print(f"라벨 생성 {len(confs)}장 / 박스 못 찾음 {len(missing)}장")
    if confs:
        print("확신도 하위 5%%: %.2f, 중앙값: %.2f" % (confs[len(confs) // 20], confs[len(confs) // 2]))

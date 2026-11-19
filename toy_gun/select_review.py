from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"C:\yolo_gun")
IMG, LBL = ROOT / "photos_jpg", ROOT / "labels_auto"
LOW_CONF, IOU_MIN = 0.6, 0.7


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


if __name__ == "__main__":
    model = YOLO(str(ROOT / "runs" / "boot" / "weights" / "best.pt"))
    files = sorted(IMG.glob("*.jpg"))
    reasons = {}
    for k in range(0, len(files), 16):
        chunk = files[k:k + 16]
        for f, r in zip(chunk, model.predict([str(p) for p in chunk], conf=0.05, verbose=False)):
            t = LBL / f"{f.stem}.txt"
            lab = [float(v) for v in t.read_text().split()[1:5]] if t.exists() and t.read_text().strip() else None
            why = []
            if lab is None:
                why.append("라벨 없음")
            if len(r.boxes) == 0:
                why.append("보조 모델도 못 찾음")
            else:
                i = int(r.boxes.conf.argmax()); conf = float(r.boxes.conf[i]); pred = r.boxes.xywhn[i].tolist()
                if conf < LOW_CONF:
                    why.append(f"확신도 {conf:.2f}")
                if lab and iou(lab, pred) < IOU_MIN:
                    why.append(f"두 모델 겹침 {iou(lab, pred):.2f}")
            if lab:
                x, y, w, h = lab
                if w * h < 0.01:
                    why.append("박스 너무 작음")
                if x - w / 2 < 0.005 or y - h / 2 < 0.005 or x + w / 2 > 0.995 or y + h / 2 > 0.995:
                    why.append("가장자리에 붙음")
            if why:
                reasons[f.stem] = why
    (ROOT / "review.txt").write_text("\n".join(reasons), encoding="utf-8")
    (ROOT / "review_reasons.txt").write_text("\n".join(f"{n}\t{', '.join(w)}" for n, w in reasons.items()), encoding="utf-8")
    from collections import Counter
    c = Counter(w.split(" ")[0] for ws in reasons.values() for w in ws)
    print(f"검수 대상 {len(reasons)}장 / 전체 {len(files)}장")
    print("이유별:", dict(c))

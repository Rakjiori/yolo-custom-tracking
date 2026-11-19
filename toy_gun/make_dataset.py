import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\yolo_gun")
IMG, LBL = ROOT / "photos_jpg", ROOT / "labels_auto"
OUT = ROOT / "dataset"
N_AUG = 4
LONG = 1024
rng = random.Random(42)


def read_box(name):
    t = (LBL / f"{name}.txt").read_text().split()
    return [float(v) for v in t[1:5]] if len(t) >= 5 else None


def write(split, stem, img, box):
    cv2.imwrite(str(OUT / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    txt = "" if box is None else "0 " + " ".join(f"{v:.6f}" for v in box) + "\n"
    (OUT / split / "labels" / f"{stem}.txt").write_text(txt)


def resize(img):
    h, w = img.shape[:2]; s = LONG / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def augment(img, box):
    h, w = img.shape[:2]
    out, b = img.copy(), list(box) if box else None
    applied = []
    if rng.random() < 0.5:
        out = cv2.flip(out, 1); applied.append("flip")
        if b: b[0] = 1 - b[0]
    if rng.random() < 0.7:
        ang = rng.uniform(-15, 15); applied.append(f"rot{ang:+.0f}")
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        if b:
            x, y, bw, bh = b[0] * w, b[1] * h, b[2] * w, b[3] * h
            pts = np.array([[x - bw/2, y - bh/2, 1], [x + bw/2, y - bh/2, 1], [x - bw/2, y + bh/2, 1], [x + bw/2, y + bh/2, 1]])
            r = pts @ M.T
            x1, y1 = np.clip(r[:, 0].min(), 0, w), np.clip(r[:, 1].min(), 0, h)
            x2, y2 = np.clip(r[:, 0].max(), 0, w), np.clip(r[:, 1].max(), 0, h)
            b = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    if rng.random() < 0.7:
        f = rng.uniform(0.75, 1.25); applied.append(f"bright{f:.2f}")
        out = cv2.convertScaleAbs(out, alpha=f, beta=0)
    r = rng.random()
    if r < 0.25:
        k = rng.choice([3, 5, 7]); out = cv2.GaussianBlur(out, (k, k), 0); applied.append(f"blur{k}")
    elif r < 0.5:
        k = rng.choice([9, 13, 17, 21]); kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kern = cv2.warpAffine(kern, M, (k, k)); kern /= kern.sum()
        out = cv2.filter2D(out, -1, kern); applied.append(f"motion{k}")
    if not applied:
        out = cv2.flip(out, 1); applied.append("flip")
        if b: b[0] = 1 - b[0]
    return out, b, applied


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    for s in ("train", "valid", "test"):
        (OUT / s / "images").mkdir(parents=True); (OUT / s / "labels").mkdir(parents=True)
    names = sorted(p.stem for p in IMG.glob("*.jpg") if (LBL / f"{p.stem}.txt").exists())
    rng.shuffle(names)
    n_val = n_test = round(len(names) * 0.1)
    split = {"valid": names[:n_val], "test": names[n_val:n_val + n_test], "train": names[n_val + n_test:]}
    log = []
    for s, items in split.items():
        for n in items:
            img = resize(cv2.imread(str(IMG / f"{n}.jpg"))); box = read_box(n)
            write(s, n, img, box)
            if s == "train":
                for k in range(N_AUG):
                    a, b, applied = augment(img, box)
                    write(s, f"{n}_aug{k + 1}", a, b); log.append(f"{n}_aug{k + 1}\t{','.join(applied)}")
    (OUT / "augment_log.txt").write_text("\n".join(log))
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['toy gun']\n")
    for s in split:
        print(f"{s:5s}: 원본 {len(split[s])}장 -> {len(list((OUT / s / 'images').glob('*.jpg')))}장")
    print("합계:", sum(len(list((OUT / s / 'images').glob('*.jpg'))) for s in split), "장")

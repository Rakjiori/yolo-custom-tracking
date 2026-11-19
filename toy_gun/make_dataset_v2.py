import random
import shutil
from pathlib import Path

import cv2

import make_dataset as md

ROOT = Path(r"C:\yolo_gun")
V1, V2 = ROOT / "dataset", ROOT / "dataset_v2"

if __name__ == "__main__":
    shutil.rmtree(V2, ignore_errors=True)
    shutil.copytree(V1, V2, ignore=shutil.ignore_patterns("*.cache"))
    md.OUT = V2

    new = sorted(p.stem for p in md.IMG.glob("neg*_*.jpg"))
    rnd = random.Random(7); rnd.shuffle(new)
    n_val = round(len(new) * 0.1)
    split = {"valid": new[:n_val], "train": new[n_val:]}
    log = []
    for s, items in split.items():
        for n in items:
            img = md.resize(cv2.imread(str(md.IMG / f"{n}.jpg"))); box = md.read_box(n)
            md.write(s, n, img, box)
            if s == "train":
                for k in range(7):
                    a, b, applied = md.augment(img, box)
                    md.write(s, f"{n}_aug{k + 1}", a, b); log.append(f"{n}_aug{k + 1}\t{','.join(applied)}")
    with open(V2 / "augment_log.txt", "a") as f:
        f.write("\n" + "\n".join(log))
    (V2 / "data.yaml").write_text(f"path: {V2.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['toy gun']\n")
    print(f"새 사진 {len(new)}장: train {len(split['train'])}장(x8) / valid {len(split['valid'])}장")
    for s in ("train", "valid", "test"):
        print(f"{s:5s}: {len(list((V2 / s / 'images').glob('*.jpg')))}장")

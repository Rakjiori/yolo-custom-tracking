import json
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"C:\yolo_gun")
RUNS = ROOT / "runs"
V1 = RUNS / "train" / "weights" / "best.pt"
V2 = RUNS / "train_v2" / "weights" / "best.pt"
VIDEOS = [ROOT / "video" / "video1.mp4", ROOT / "video" / "video3.mov"]


def track(weights, video, save=None):
    m = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = det = 0; confs, ids, per_frame = [], Counter(), []
    frames = []
    for r in m.track(str(video), tracker=str(ROOT / "bytetrack_tuned.yaml"), conf=0.25, imgsz=960,
                     stream=True, persist=True, verbose=False):
        n += 1; k = len(r.boxes); per_frame.append(k)
        if k: det += 1; confs += r.boxes.conf.tolist()
        if r.boxes.id is not None: ids.update(r.boxes.id.int().tolist())
        if save:
            im = r.plot(line_width=5); h, w = im.shape[:2]; s = 1380 / max(h, w)
            frames.append(cv2.resize(im, (int(w * s) // 2 * 2, int(h * s) // 2 * 2))[:, :, ::-1])
    if save:
        imageio.mimwrite(save.with_suffix(".mp4"), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        t = [cv2.resize(frames[i][:, :, ::-1], (270, 480)) for i in idx]
        cv2.imwrite(str(save.with_name(save.stem + "_frames.jpg")), np.vstack([np.hstack(t[:4]), np.hstack(t[4:])]))
    return {"frames": n, "detected_frames": det, "multi_box_frames": sum(1 for k in per_frame if k > 1),
            "mean_conf": round(float(np.mean(confs)), 3) if confs else 0, "num_ids": len(ids), "ids": dict(ids)}


if __name__ == "__main__":
    model = YOLO(r"C:\yolo_project\yolo11s.pt")
    model.train(data=str(ROOT / "dataset_v2" / "data.yaml"), epochs=80, imgsz=640, batch=16, device=0,
                workers=4, patience=20, project=str(RUNS), name="train_v2", exist_ok=True)

    m = YOLO(str(V2)).val(data=str(ROOT / "dataset_v2" / "data.yaml"), split="test", plots=False, verbose=False)
    test = {"images": 83, "p": float(m.box.mp), "r": float(m.box.mr), "map50": float(m.box.map50), "map": float(m.box.map)}
    (RUNS / "test_metrics_v2.json").write_text(json.dumps(test, indent=2))
    print("\n=== v2 test ===", test)

    out = RUNS / "demo_v2"; out.mkdir(exist_ok=True)
    result = {}
    for v in VIDEOS:
        result[v.stem] = {"v1": track(V1, v, out / f"{v.stem}_v1"), "v2": track(V2, v, out / f"{v.stem}_v2")}
        print(v.stem, json.dumps(result[v.stem], ensure_ascii=False))
    (out / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

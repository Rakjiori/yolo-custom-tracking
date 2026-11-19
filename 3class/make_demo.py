import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from ultralytics import YOLO

PROJECT = Path(r"C:\yolo_project")
BEST = PROJECT / "runs" / "train" / "weights" / "best.pt"
OUT = PROJECT / "runs" / "demo"
TOTAL_SEC = 7.0
CONF = 0.4


def track_video(model, path):
    frames, info = [], []
    fps = None
    for r in model.track(source=str(path), tracker="bytetrack.yaml", conf=CONF,
                         stream=True, persist=True, verbose=False):
        frames.append(r.plot()[:, :, ::-1])
        cls = [model.names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        ids = r.boxes.id.int().tolist() if r.boxes is not None and r.boxes.id is not None else []
        info.append({"classes": set(cls), "n": len(cls), "ids": ids})
    import cv2
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or 24.0
    return frames, info, fps


def best_window(info, length):
    score = [len(i["classes"]) for i in info]
    if len(score) <= length:
        return 0
    sums = [sum(score[s:s + length]) for s in range(len(score) - length + 1)]
    return int(np.argmax(sums))


if __name__ == "__main__":
    videos = [Path(v) for v in sys.argv[1:]]
    if not videos:
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)

    per_video_sec = TOTAL_SEC / len(videos)
    clips, fps_out = [], None
    for v in videos:
        model = YOLO(str(BEST))
        frames, info, fps = track_video(model, v)
        fps_out = fps_out or fps
        imageio.mimwrite(OUT / f"{v.stem}_tracked.mp4", frames, fps=fps,
                         codec="libx264", quality=8, macro_block_size=1)

        all3 = sum(len(i["classes"]) == 3 for i in info)
        ids = sorted({i for f in info for i in f["ids"]})
        cnt = Counter(c for f in info for c in f["classes"])
        print(f"\n[{v.name}] {len(frames)} 프레임, {len(frames)/fps:.1f}초")
        print(f"  세 클래스 모두 탐지된 프레임: {all3}/{len(frames)} ({all3/len(frames):.0%})")
        print(f"  클래스별 탐지 프레임 수: {dict(cnt)}")
        print(f"  사용된 추적 ID: {ids}")

        n = int(round(per_video_sec * fps))
        s = best_window(info, n)
        print(f"  사용 구간: {s/fps:.2f}s ~ {(s+n)/fps:.2f}s")
        clips.extend(frames[s:s + n])

    out = OUT / "result.mp4"
    imageio.mimwrite(out, clips, fps=fps_out, codec="libx264", quality=8, macro_block_size=1)
    print(f"\n최종 영상: {out} ({len(clips)/fps_out:.1f}초)")

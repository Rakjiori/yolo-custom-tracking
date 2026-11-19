import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(r"C:\yolo_project")
BEST = PROJECT / "runs" / "train" / "weights" / "best.pt"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("사용법: python track.py 영상파일.mp4")

    model = YOLO(str(BEST))
    model.track(
        source=sys.argv[1],
        tracker="bytetrack.yaml",
        conf=0.4,
        save=True,
        project=str(PROJECT / "runs"),
        name="track",
        exist_ok=True,
    )
    print(f"결과 영상 폴더: {PROJECT / 'runs' / 'track'}")

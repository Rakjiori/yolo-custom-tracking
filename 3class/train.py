from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(r"C:\yolo_project")
DATA = PROJECT / "dataset" / "data.yaml"

if __name__ == "__main__":
    if not DATA.exists():
        raise SystemExit(f"data.yaml 이 없습니다: {DATA}\nRoboflow 데이터셋을 먼저 압축 해제하세요.")

    model = YOLO(str(PROJECT / "yolo11s.pt"))
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        project=str(PROJECT / "runs"),
        name="train",
        exist_ok=True,
    )

    best = PROJECT / "runs" / "train" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA), split="test")
    print("\n=== test 성능 ===")
    print(f"Precision : {metrics.box.mp:.3f}")
    print(f"Recall    : {metrics.box.mr:.3f}")
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")
    print(f"모델 파일 : {best}")

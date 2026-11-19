from pathlib import Path

ROOT = Path(r"C:\yolo_gun")


if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO(r"C:\yolo_project\yolo11s.pt")
    model.train(data=str(ROOT / "dataset" / "data.yaml"), epochs=80, imgsz=640, batch=16,
                device=0, workers=4, patience=20, project=str(ROOT / "runs"), name="train", exist_ok=True)
    best = ROOT / "runs" / "train" / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(ROOT / "dataset" / "data.yaml"), split="test")
    print("\n=== test 성능 ===")
    print(f"Precision : {m.box.mp:.3f}\nRecall    : {m.box.mr:.3f}\nmAP50     : {m.box.map50:.3f}\nmAP50-95  : {m.box.map:.3f}")
    print(f"모델 파일 : {best}")

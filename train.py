#训练
from pathlib import Path

from ultralytics import YOLO
import ultralytics.nn.tasks

ROOT = Path(__file__).resolve().parent
model = YOLO(str(ROOT / "yaml" / "PC2f_MPF_yolov8s.yaml"))
results = model.train(
    data=str(ROOT / "data" / "escvehicle.yaml"),
    batch=16,
    epochs=200,
    imgsz=704,
    device="4,5,6,7",
)

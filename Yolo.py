from ultralytics import YOLO
model = YOLO('best.pt')

results = model(source="Vid.mp4",save=True,data='data.yaml')
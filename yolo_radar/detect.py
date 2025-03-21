from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("best.pt")

result = model.predict(source = "DEMO.mp4", show=True, save=True, line_width = 2, save_txt = True,conf=0.7
)
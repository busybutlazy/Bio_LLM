from ultralytics import YOLO

model=YOLO("/app/runs/detect/train12/weights/best.pt")

results=model("/app/datas/test_imgs/113學測-1.png",save=True)

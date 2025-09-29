import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = "best.onnx"  # Change this to your actual model file
model = YOLO(model_path)

# Load an image or video for inference
source = "/Users/nguyenconghung/Downloads/image_train/04_17-27-08.jpg"  # Change to your image or video file, or use 0 for webcam

# Perform inference
results = model(source)  # Can be an image, video file, or directory

# Process and display results
for result in results:
    img = result.plot()  # Get image with detections
    cv2.imshow("YOLOv8 Inference", img)
    cv2.waitKey(0)  # Press any key to close

cv2.destroyAllWindows()

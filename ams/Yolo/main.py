import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import math
from ultralytics import YOLO

def crop_bbox(image, boxes):
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
    return cropped_images

def crop_faces_with_yolo_haar(image_path):
    frame = cv2.imread(image_path)
    my_model = YOLO('./Yolo/best.pt')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    result = my_model(frame, conf=0.15)[0]
    boxes = result.boxes
    cropped_images = crop_bbox(frame, boxes)
    output_folder = "./faces"
    os.makedirs(output_folder, exist_ok=True)
    for i, cropped_image in enumerate(cropped_images):
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            for j, (x, y, w, h) in enumerate(faces):
                cropped_face = cropped_image[y:y+h, x:x+w]
                output_path = os.path.join(output_folder, f"face_{i}_{j}.jpg")
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved: {output_path}")
        else:
            output_path = os.path.join(output_folder, f"face_{i}_no_face_detected.jpg")
            cv2.imwrite(output_path, cropped_image)
            print(f"No face detected in cropped image, saved: {output_path}")
    return output_path

# source = '/content/Screenshot_20240503-211938.png'

# output_folder = "faces_yolo_haar"



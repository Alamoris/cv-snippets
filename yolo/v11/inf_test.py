import os
from pathlib import Path
import traceback

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)


model_path = "../../data/weights/yolo11m.pt"
model = YOLO(model_path)

classes = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def main(video_file):
    cap = cv2.VideoCapture(str(video_file))

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video')
            break

        results = model.track(frame, classes=[0, 2], persist=True)

        annotated_frame = results[0].plot()

        annotated_frame = cv2.resize(annotated_frame, (1920, 1080))
        cv2.imshow("Frame", annotated_frame)

        if cv2.waitKey(1) == ord('q'):  # Exit on 'q'
            break
        frame_id += 1


if __name__ == "__main__":
    video_name = Path('office_camera_test.mp4')
    source_path = Path('../../data/videos/')
    video_file = source_path / video_name

    try:
        main(video_file)
    except Exception as e:
        print('Exception during video showing: ', e)
        traceback.print_exc()

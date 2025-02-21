import json
import csv
from pathlib import Path

import numpy as np
import cv2

from ultralytics import YOLO
import supervision as sv


# Model
# model_path = "../WALDO-master/WALDO30_yolov8n_640x640.pt"
# model_path = "../WALDO-master/yolov8n.pt"
model_path = "../WALDO-master/yolov8m-worldv2.pt"
model = YOLO(model_path)


box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)

# classNames = ['LightVehicle', 'Person', 'Building', 'UPole', 'Container', 'Boat', 'Bike', 'Container', 'Truck', 'Gastank', 'Digger', 'Solarpanels', 'Bus']
# allowed_classes = ['LightVehicle', 'Person', 'Building', 'UPole', 'Container', 'Boat', 'Bike', 'Container', 'Truck', 'Gastank', 'Digger', 'Solarpanels', 'Bus']
# allowed_classes = ['LightVehicle', 'Person', 'Boat']
# allowed_ids = [classNames.index(al_class) for al_class in allowed_classes]


classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
allowed_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def draw_image(image, classes, confidences, box_coords):
    if classes:
        try:
            class_ids, confidences, xyxy = zip(
                *((cls, conf, xy) for cls, conf, xy in zip(classes, confidences, box_coords) if
                    cls in allowed_ids))
        except ValueError as exc:
            class_ids, confidences, xyxy = [], [], []

        print('class_ids', classes, confidences, box_coords, allowed_ids)
        if class_ids:
            class_ids = np.array(class_ids, dtype=int)
            confidences = np.array(confidences, dtype=np.float32)
            xyxy = np.array(xyxy, dtype=np.float32)

            class_names = [classNames[cls] for cls in class_ids]

            # Create a sv.Detections object for annotation
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids
            )

            # Prepare labels for label annotator
            labels = [f"{class_name}" for class_name in class_names]

            # Annotate the image
            image = box_annotator.annotate(scene=image, detections=detections)
            image = label_annotator.annotate(scene=image, detections=detections, labels=labels)

    return image


def process_video(video_path, csv_writer):
    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0

    while True:
        # if frame_id > 150:
        #     break

        success, img = cap.read()
        frame = img.copy()
        if not success:  # Exit loop if no more frames
            print(f"End of video file or error reading file: {video_path}")
            break

        results = model(img, stream=True)
        # results = model(img)

        res = {
            'frame_id': frame_id,
            'classes': [],
            'confidences': [],
            'box_coords': [],
        }
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                res['classes'].append(classNames[int(box.cls[0])])
                res['confidences'].append(round(float(box.conf[0]), 2))
                res['box_coords'].append([int(x1), int(y1), int(x2), int(y2)])

        # frame = draw_image(frame, res['classes'], res['confidences'], res['box_coords'])
        # frame = cv2.resize(frame, (int(frame.shape[1] / 1.5), int(frame.shape[0] / 1.5)))

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(0) == ord('q'):
        #     break

        res['classes'] = json.dumps(res['classes'])
        res['confidences'] = json.dumps(res['confidences'])
        res['box_coords'] = json.dumps(res['box_coords'])
        csv_writer.writerow(res)
        csv_file.flush()

        frame_id += 1

    cap.release()


def prepare_csv(csv_file_name):
    fieldnames = [
        'frame_id',
        'classes',
        'confidences',
        'box_coords',
        'deep_sort_ids',
        'bytetrack_ids'
    ]

    csv_file = open(csv_file_name, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    return csv_file, csv_writer


if __name__ == "__main__":
    source_path = Path('./videos/')
    video_file = source_path / 'REC_yolov8n_test.mp4'

    csv_file, csv_writer = prepare_csv(source_path / (video_file.stem + '.csv'))

    try:
        process_video(video_file, csv_writer)
    except Exception as e:
        print(e)
    finally:
        csv_file.close()

    cv2.destroyAllWindows()

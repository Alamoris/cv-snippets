import csv
import json
import sys
import time
from pathlib import Path
import traceback

import cv2
import numpy as np
import supervision as sv
from Cython import returns
from scipy.optimize import brent

from supervision.annotators.utils import ColorLookup
from supervision.draw.color import Color, ColorPalette

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
# allowed_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# classNames = ['LightVehicle', 'Person', 'Building', 'UPole', 'Container', 'Boat', 'Bike', 'Container', 'Truck', 'Gastank', 'Digger', 'Solarpanels', 'Bus']
# allowed_classes = ['LightVehicle', 'Person', 'Building', 'UPole', 'Container', 'Boat', 'Bike', 'Container', 'Truck', 'Gastank', 'Digger', 'Solarpanels', 'Bus']
allowed_classes = ['person', 'car']
allowed_ids = [classNames.index(al_class) for al_class in allowed_classes]

# new_class_map = {
#     'car': 1,
#     'person': 2,
# }

new_class_map = {
    'car': 0,
    'suv': 1,
    'land vehicle': 2,
    'van': 3,
    'person': 4,
    'sneak': 5,
    'man': 6,
    'woman': 7,
    'bicycle': 8,
    'motorcycle': 9,
    'vehicle registration plate': 10,
}


box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)


def main(video_file, csv_reader, space_controll=False, jump_to_frame=0, conf_threshold=0.4, draw_ids=True):
    cap = cv2.VideoCapture(str(video_file))
    cap.set(cv2.CAP_PROP_FPS, 30.25)

    frame_id = 0

    if jump_to_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to_frame)
        frame_id = jump_to_frame

        for i in range(jump_to_frame):
            # _, _ = cap.read()
            csv_reader.__next__()

    print('number of frames', cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video')
            break

        row = csv_reader.__next__()

        class_data = json.loads(row[1])
        if class_data:
            confidences_data = json.loads(row[2])
            xyxy_data = json.loads(row[3])
            ds_ids = json.loads(row[4])
            bt_ids = json.loads(row[5])

            try:
                class_ids, class_names, confidences, xyxy, dsids, btids = zip(
                    *((classNames.index(cls), cls, conf, xy, dsid, btid) for cls, conf, xy, dsid, btid in zip(class_data, confidences_data, xyxy_data, ds_ids, bt_ids) if
                        cls in allowed_classes and conf > conf_threshold))

                labels = [f"{class_name} #{id}" for class_name, id in zip(class_names, dsids)]

                # dsid_data = json.loads(row[4])
                # class_ids = []
                # labels = []
                # xyxy = []
                # confidences = []
                # for id, class_n in enumerate(class_data):
                #     clid = new_class_map.get(class_n)
                #     if clid is not None:
                #         class_ids.append(clid)
                #         labels.append(f'{class_n} #{dsid_data[id]}')
                #         xyxy.append(xyxy_data[id])
                #         confidences.append(confidences_data[id])
            except ValueError as exc:
                class_ids, confidences, xyxy = [], [], []

            if class_ids:
                class_ids = np.array(class_ids, dtype=int)
                confidences = np.array(confidences, dtype=np.float32)
                xyxy = np.array(xyxy, dtype=np.float32)

                # Create a sv.Detections object for annotation
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidences,
                    class_id=class_ids
                )

                # Annotate the image
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        print('frame_id', frame_id)
        frame = cv2.resize(frame, (round(frame.shape[1] / 1.5), round(frame.shape[0] / 1.5)))
        cv2.imshow('frame', frame)

        if space_controll:
            if cv2.waitKey(0) == ord('q'):  # Exit on 'q'
                break
        else:
            if cv2.waitKey(1) == ord('q'):  # Exit on 'q'
                break
        frame_id += 1


def prepare_csv_reader(csv_path):
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    r = csv_reader.__next__()  # Read headers
    return csv_file, csv_reader


if __name__ == "__main__":
    jump_to_frame = 0
    source_path = Path('../data/videos/')
    video_name = Path('DJI_0174.MP4')
    video_file = source_path / video_name
    csv_file, csv_reader = prepare_csv_reader(source_path / (video_name.stem + '.csv'))

    # sys.exit()
    try:
        print('jump_to_frame', jump_to_frame)
        main(video_file, csv_reader, space_controll=True, jump_to_frame=jump_to_frame)
    except Exception as e:
        print('Exception during video showing: ', e)
        traceback.print_exc()
    finally:
        csv_file.close()

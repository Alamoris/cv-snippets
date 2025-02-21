import cv2
import csv
import json
import traceback
from pathlib import Path
import logging
import time

import numpy as np
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

logging.basicConfig(level=logging.INFO)

deep_sort = DeepSort(max_age=5)

box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)


class_map = {
    'helmet': 0,
    'glasses': 1,
    'handbag': 2,
    'car tire': 3,
    'suv': 4,
    'vehicle registration plate': 5,
    'camera': 6,
    'building': 7,
    'desk': 8,
    'chair': 9,
    'person': 10,
    'human face': 11,
    'footwear': 12,
    'van': 13,
    'bicycle wheel': 14,
    'sneakers': 15,
    'street light': 16,
    'hat': 17,
    'wheel': 18,
    'backpack': 19,
    'baseball glove': 20,
    'skateboard': 21,
    'bicycle': 22,
    'truck': 23,
    'car': 24,
    'bus': 25,
    'motorcycle': 26,
    'leather shoes': 27,
    'trousers': 28,
    'land vehicle': 29,
    'jacket': 30,
    'sneaker': 31,
    'walking shoe': 32,
    'boots': 33,
    'trash bin/can': 34
}

reverse_class_map = {v: k for k, v in class_map.items()}
allowded_classes = ['car', 'suv', 'land vehicle', 'van', 'person', 'sneak', 'man', 'woman', 'bicycle', 'motorcycle', 'vehicle registration plate']

from scipy.optimize import linear_sum_assignment

def _transfer_bounding_boxes_hungarian(prev_tracked: sv.Detections,
                                       new_detections: sv.Detections,
                                       max_dist=150.0) -> sv.Detections:
    old_xyxy = prev_tracked.xyxy  # shape (N_old,4)
    old_ids = prev_tracked.tracker_id  # shape (N_old,)
    new_xyxy = new_detections.xyxy  # shape (N_new, 4)

    n_new = len(new_xyxy)
    n_old = len(old_xyxy)

    # If we have no old tracks, just return new detections with None IDs
    if n_old == 0:
        tracker_id = np.array([None] * n_new, dtype=object)
        return sv.Detections(
            xyxy=new_xyxy.copy(),
            confidence=new_detections.confidence.copy(),
            class_id=new_detections.class_id.copy(),
            tracker_id=tracker_id
        )

    # Build cost matrix (squared center distance)
    cost_matrix = np.zeros((n_new, n_old), dtype=np.float32)
    for i in range(n_new):
        x1, y1, x2, y2 = new_xyxy[i]
        n_cx, n_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        for j in range(n_old):
            ox1, oy1, ox2, oy2 = old_xyxy[j]
            o_cx, o_cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            dist_sq = (n_cx - o_cx) ** 2 + (n_cy - o_cy) ** 2
            cost_matrix[i, j] = dist_sq

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    tracker_id = np.array([None] * n_new, dtype=object)
    max_dist_sq = max_dist ** 2
    for (r, c) in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= max_dist_sq:
            tracker_id[r] = old_ids[c]

    return sv.Detections(
        xyxy=new_xyxy.copy(),
        confidence=new_detections.confidence.copy(),
        class_id=new_detections.class_id.copy(),
        tracker_id=tracker_id
    )


def main(video_file, csv_reader, csv_writer):
    cap = cv2.VideoCapture(str(video_file))
    frame_id = 0

    print('number of frames', cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS))


    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video')
            break

        row = csv_reader.__next__()

        anno_class_names = json.loads(row[1])
        anno_conf = json.loads(row[2])
        if not anno_conf:
            anno_conf = np.zeros(len(json.loads(row[1])))
        final_boxes = json.loads(row[3])

        try:
            anno_class_names, anno_class_ids, anno_conf, final_boxes = zip(
                *((cls, class_map[cls], conf, xy) for cls, conf, xy in zip(anno_class_names, anno_conf, final_boxes) if
                    cls in allowded_classes))
        except Exception:
            anno_class_names = []
            anno_conf = []
            final_boxes = []

        matched_ds_ids = []

        print(f'frame_id, {frame_id}')
        print(anno_class_names, anno_class_ids, anno_conf, final_boxes)

        if anno_class_names:
            deep_sort_detections = []
            for (x1, y1, x2, y2), conf, c_id in zip(final_boxes, anno_conf, anno_class_ids):
                w, h = x2 - x1, y2 - y1
                deep_sort_detections.append(([x1, y1, w, h], conf, c_id))

            ds_tracks = deep_sort.update_tracks(deep_sort_detections, frame=frame)

            # Convert DS tracks to [x1, y1, x2, y2], track_id
            ds_bboxes = []
            ds_ids = []
            for track in ds_tracks:
                if not track.is_confirmed():
                    continue
                x1t, y1t, x2t, y2t = track.to_ltrb()
                ds_bboxes.append([x1t, y1t, x2t, y2t])
                ds_ids.append(track.track_id)

            # Ensure ds_bboxes is 2D (N,4)
            if len(ds_bboxes) == 0:
                ds_bboxes_arr = np.empty((0, 4), dtype=np.float32)
            else:
                ds_bboxes_arr = np.array(ds_bboxes, dtype=np.float32)

            ds_new = sv.Detections(
                xyxy=np.array(final_boxes, dtype=np.float32),
                confidence=np.array(anno_conf, dtype=np.float32),
                class_id=np.array(anno_class_ids, dtype=int),
                tracker_id=None
            )

            ds_matched = _transfer_bounding_boxes_hungarian(
                prev_tracked=sv.Detections(
                    xyxy=ds_bboxes_arr,
                    confidence=None,
                    class_id=None,
                    tracker_id=np.array(ds_ids, dtype=object)
                ),
                new_detections=ds_new
            )

            matched_ds_ids = ds_matched.tracker_id
            if matched_ds_ids is None:
                matched_ds_ids = np.array([], dtype=object)

            matched_ds_ids = list(map(lambda dsid: 0 if dsid is None else int(dsid), matched_ds_ids))
            print('matched_ds_ids', matched_ds_ids)

            # labels = [f'{reverse_class_map[class_id]} #{dsid}' for class_id, dsid in zip(anno_class_ids, matched_ds_ids)]

            # detections = sv.Detections(
            #     xyxy=np.array(final_boxes),
            #     confidence=np.array(anno_conf),
            #     class_id=np.array(anno_class_ids),
            # )

            # # Annotate the image
            # frame = box_annotator.annotate(scene=frame, detections=detections)
            # frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        res = {
            'frame_id': row[0],
            'classes': json.dumps(anno_class_names),
            'confidences': json.dumps(anno_conf),
            'box_coords': json.dumps(final_boxes),
            'bytetrack_ids': json.dumps(matched_ds_ids),
        }
        csv_writer.writerow(res)
        frame_id += 1

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(0) == ord('q'):
        #     break


def get_unique_classes(csv_reader):
    unique_classes = set()
    try:
        while True:
            row = csv_reader.__next__()

            class_names = json.loads(row[1])
            unique_classes.update(class_names)
    finally:
        print(dict(zip(unique_classes, range(len(unique_classes)))))


def morpth_ids(csv_reader, csv_writer):
    # replacement_map = {  # demo_3
    #     4: 1, 8: 1, 53: 1, 715 : 1,
    #     6: 2, 7: 2, 149: 2, 712: 2,
    #     5: 3, 9: 3,
    #     716: 2,
    #     3: 1,
    # }

    # replacement_map = {  # demo_4
    #     1: 1,
    #     5: 1, 8: 1, 9: 1, 10: 1, 20: 1, 41: 1,
    #     19: 2, 25: 2, 40: 2, 50: 2, 51: 2, 53: 2, 54: 2,
    #     56: 3, 62: 3, 66: 3, 67: 3, 70: 3, 80: 3, 
    #     64: 4,
    #     73: 5,
    #     75: 6,
    #     78: 7, 79: 7,
    #     82: 8,
    #     11: 999
    # }

    replacement_map = {  # REC_yolov8n_test_cut
        112: 1, 211: 1,
        185: 2, 174: 3, 150: 4, 44: 5, 27: 6, 9: 7,
        6: 8, 238: 9, 244: 10, 245: 11, 240: 12, 239: 13, 243: 14,
        253: 4, 246: 15, 264: 14
    }

    try:
        while True:
            row = csv_reader.__next__()
            bytetrack_ids = json.loads(row[4])

            new_bytetrack_ids = list(map(lambda btid: replacement_map[btid] if btid in replacement_map else btid, bytetrack_ids))

            res = {
                'frame_id': row[0],
                'classes': row[1],
                'confidences': row[2],
                'box_coords': row[3],
                'bytetrack_ids': json.dumps(new_bytetrack_ids),
            }
            csv_writer.writerow(res)
    except StopIteration:
        print('end of file')


def clean_classes(csv_reader, csv_writer):
    allowded_ids = [1, 2, 3]

    try:
        while True:
            row = csv_reader.__next__()
            classes = json.loads(row[1])
            confidences = json.loads(row[2])
            box_coords = json.loads(row[3])
            bytetrack_ids = json.loads(row[4])

            try:
                classes, confidences, box_coords, bytetrack_ids = zip(
                    *((cls, conf, xy, bti) for cls, conf, xy, bti in zip(classes, confidences, box_coords, bytetrack_ids) if
                        bti in allowded_ids))
            except Exception:
                classes = []
                confidences = []
                box_coords = []
                bytetrack_ids = []

            res = {
                'frame_id': row[0],
                'classes': json.dumps(classes),
                'confidences': json.dumps(confidences),
                'box_coords': json.dumps(box_coords),
                'bytetrack_ids': json.dumps(bytetrack_ids),
            }
            csv_writer.writerow(res)
    except StopIteration:
        print('end of file')


def fill_dummy_ids(csv_reader, csv_writer):
    try:
        while True:
            row = csv_reader.__next__()
            classes = json.loads(row[1])
            res = {
                'frame_id': row[0],
                'classes': row[1],
                'confidences': row[2],
                'box_coords': row[3],
                'bytetrack_ids': json.dumps(np.ones(len(classes)).tolist()),
            }
            csv_writer.writerow(res)
    except StopIteration:
        print('end of file')


def prepare_csv_reader(csv_path):
    csv_file = open(csv_path, 'r')
    # for i in range(98):
    #     csv_file.readline()
    csv_reader = csv.reader(csv_file, delimiter=',')
    r = csv_reader.__next__()  # Read headers
    return csv_file, csv_reader


def prepare_csv_writer(csv_path):
    fieldnames = ['frame_id', 'classes', 'confidences', 'box_coords', 'bytetrack_ids']

    csv_writer_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_writer_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    return csv_writer_file, csv_writer


if __name__ == '__main__':
    video_name = Path('REC_yolov8n_test_cut.mp4')
    source_path = Path('./videos/')
    video_file = source_path / video_name

    csv_reader_file, csv_reader = prepare_csv_reader(source_path / (video_name.stem + '.csv'))

    # sys.exit()
    try:
        # csv_writer_file, csv_writer = prepare_csv_writer(source_path / f'{video_name.stem}_dsed.csv')
        # main(video_file, csv_reader, csv_writer)

        csv_writer_file, csv_writer = prepare_csv_writer(source_path / f'{video_name.stem}_m.csv')
        morpth_ids(csv_reader, csv_writer)

        # csv_writer_file, csv_writer = prepare_csv_writer(source_path / f'{video_name.stem}_fin.csv')
        # clean_classes(csv_reader, csv_writer)

        # csv_writer_file, csv_writer = prepare_csv_writer(source_path / f'{video_name.stem}_dummy_ids.csv')
        # fill_dummy_ids(csv_reader, csv_writer)

        # get_unique_classes(csv_reader)
    except Exception as e:
        print('Exception during video showing: ', e)
        traceback.print_exc()
    finally:
        csv_reader_file.close()
        csv_writer_file.close()

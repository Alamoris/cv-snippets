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

    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('number of frames', n_frames, cap.get(cv2.CAP_PROP_FPS))

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
        bs_ids = json.loads(row[5])

        try:
            anno_class_names, anno_class_ids, anno_conf, final_boxes, bsids = zip(
                *((cls, class_map[cls], conf, xy, bs_id) for cls, conf, xy, bs_id in zip(anno_class_names, anno_conf, final_boxes, bs_ids) if
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
            print('deep_sort_detections', deep_sort_detections)

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
            'bytetrack_ids': json.dumps(bsids),
            'deep_sort_ids': json.dumps(matched_ds_ids),
        }
        csv_writer.writerow(res)
        frame_id += 1
        print(f'Frame: {frame_id}/{n_frames}')


        # cv2.imshow('frame', frame)
        # if cv2.waitKey(0) == ord('q'):
        #     break


def prepare_csv_reader(csv_path):
    csv_file = open(csv_path, 'r')
    # for i in range(98):
    #     csv_file.readline()
    csv_reader = csv.reader(csv_file, delimiter=',')
    r = csv_reader.__next__()  # Read headers
    return csv_file, csv_reader


def prepare_csv_writer(csv_path):
    fieldnames = ['frame_id', 'classes', 'confidences', 'box_coords', 'deep_sort_ids', 'bytetrack_ids']

    csv_writer_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_writer_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    return csv_writer_file, csv_writer


if __name__ == '__main__':
    video_name = Path('DJI_0173.MP4')
    source_path = Path('../data/videos/')
    video_file = source_path / video_name

    csv_reader_file, csv_reader = prepare_csv_reader(source_path / (video_name.stem + '.csv'))

    # sys.exit()
    try:
        csv_writer_file, csv_writer = prepare_csv_writer(source_path / f'{video_name.stem}_dsed.csv')
        main(video_file, csv_reader, csv_writer)
    except Exception as e:
        print('Exception during video showing: ', e)
        traceback.print_exc()
    finally:
        csv_reader_file.close()
        csv_writer_file.close()

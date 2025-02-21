import cv2
import sys
import json
import time
import numpy as np
from pathlib import Path
import logging

sys.path.append('/home/alamoris/data/skydrones/code/cv-snippets')

import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.csv_helpers import prepare_csv_reader


logging.basicConfig(level=logging.INFO)


box_annotator = sv.BoxCornerAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)


allowed_classes = ['person', 'car']
COLORS = np.random.randint(0, 255, size=(len(allowed_classes), 3))


def annotate(tracks, frame, resized_frame, frame_width, frame_height, colors):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_class = track.det_class
        x1, y1, x2, y2 = track.to_ltrb()
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        # Annotate boxes.
        color = colors[int(track_class)]
        cv2.rectangle(
            frame,
            p1,
            p2,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=2
        )
        # Annotate ID.
        cv2.putText(
            frame, f"ID: {track_id}",
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )
    return frame


def process(tracker, frame, row):
    classes = json.loads(row[1])
    confs = json.loads(row[2])
    boxes = json.loads(row[3])

    deep_sort_detections = []
    for (x1, y1, x2, y2), conf, c_name in zip(boxes, confs, classes):
        w, h = x2 - x1, y2 - y1
        deep_sort_detections.append(([x1, y1, w, h], conf, allowed_classes.index(c_name)))

    track_start_time = time.time()
    tracks = tracker.update_tracks(deep_sort_detections, frame=frame)
    track_end_time = time.time()
    track_fps = 1 / (track_end_time - track_start_time)

    logging.info(f"Tracking FPS: {track_fps:.1f}, Total FPS: {track_fps:.1f}")

    # Draw bounding boxes and labels on frame.
    if len(tracks) > 0:
        frame = annotate(
            tracks,
            frame,
            frame,
            0,
            0,
            COLORS
        )

    return frame


def main(video_path, csv_reader):
    cap = cv2.VideoCapture(str(video_path))

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f'Video info: {frame_count}, FPS: {fps}, file_name: {video_path}')

    tracker = DeepSort(max_age=30)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video')
            break

        data_row = next(csv_reader)

        frame = process(tracker, frame, data_row)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == ord('q'):  # Exit on 'q'
            break


    logging.info(f'Frame id: {frame_id}')
    frame_id += 1


if __name__ == "__main__":
    video_name = Path('office_camera_test.mp4')
    source_path = Path('../data/videos/')

    video_file = source_path / video_name
    csv_file, csv_reader = prepare_csv_reader(source_path / (video_name.stem + '.csv'))

    try:
        main(video_file, csv_reader)
    finally:
        csv_file.close()

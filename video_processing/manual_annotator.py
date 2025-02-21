import csv
import json
import sys
import os
from datetime import datetime
from pathlib import Path
import traceback

import cv2
import numpy as np
import supervision as sv


FLORENCE_ID_MAP = {
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
    'inc': 199,
    'act': 200,
}

FLORENCE_REVERSE_ID_MAP = {v: k for k, v in FLORENCE_ID_MAP.items()}


class DispatchDemoAnnotator:
    def __init__(self, video_name, source_path, csv_data_prefix='', start_frame=0, end_csv=True, florence=True, trace_ids=[]):
        self.start_frame = start_frame
        self.end_csv = end_csv
        self.florence = florence
        self.trace_ids = trace_ids

        self.source_path = source_path
        self.video_path = source_path / video_name

        if csv_data_prefix == 'last':
            # dirlist = os.listdir('./video_streaming_demo/videos/')
            dirlist = os.listdir('./videos/')

            only_manual_markup = list(filter(lambda name: name.startswith(video_name.stem + '_') and name.endswith('.csv'), dirlist))

            last_date = None
            last_date_string = None
            for markup_name in only_manual_markup:
                sting_time = markup_name[len(video_name.stem)+1:len(markup_name)-4]
                date = datetime.strptime(sting_time, "%m-%d-%Y:%H-%M-%S")

                if last_date is None:
                    last_date = date
                    last_date_string = sting_time
                    continue

                if date > last_date:
                    last_date = date
                    last_date_string = sting_time

            if last_date_string is None:
                print('There is no manual marking yet')
                sys.exit(0)

            csv_data_prefix = f'_{last_date_string}'

        self.prepare_csv_reader_writer(source_path, video_name.stem, csv_prefix=csv_data_prefix)
        self.prepare_video()

        self.classNames = ['LightVehicle', 'Person', 'Building', 'UPole', 'Container', 'Boat', 'Bike', 'Container', 'Truck',
                           'Gastank', 'Digger', 'Solarpanels', 'Bus']
        allowed_classes = ['LightVehicle', 'Person', 'Boat']
        self.allowed_ids = [self.classNames.index(al_class) for al_class in allowed_classes]

        self.annotated_frame = None
        self.raw_frame = None
        self.frame_id = 0
        self.classes = np.array([], int)
        self.confidences = np.array([])
        self.boxes_coordinates = np.empty((0, 2))
        self.bytetrack_ids = np.array([])

        self.del_classes = np.array([], int)
        self.del_confidences = np.array([])
        self.del_boxes_coordinates = np.empty((0, 2))
        self.del_bytetrack_ids = np.array([])

        self.trace_coords = []
        self.trace_classes = []

        if self.florence:
            self.append_class = FLORENCE_ID_MAP['inc']
        else:
            self.append_class = 1
        self.cap_width = None
        self.cap_height = None
        self.cap_frame_num = None
        self.event_handle_type = 0  # 0 to delete rect 1 to append rect
        self.select_rect = False
        self.left_mouse_state = False
        self.drawing_start = (None, None)

        self.clip = lambda mx, mn, val: max(mn, min(val, mx))

    def cleanup(self):
        self.csv_reader_file.close()
        self.csv_writer_file.close()

    def finalize_csv(self):
        res = {
            'frame_id': self.frame_id,
            'classes': json.dumps(self.classes.tolist() if not self.florence else [FLORENCE_REVERSE_ID_MAP[cls] for cls in self.classes]),
            'confidences': json.dumps(self.confidences.tolist()),
            'box_coords': json.dumps(self.boxes_coordinates.tolist()),
        }
        if self.florence:
            res['bytetrack_ids'] = json.dumps(self.bytetrack_ids.tolist())
        self.csv_writer.writerow(res)

        for row in self.csv_reader:
            res = {
                'frame_id': row[0],
                'classes': row[1],
                'confidences': row[2],
                'box_coords': row[3],
            }
            if self.florence:
                try:
                    res['bytetrack_ids'] = row[4]
                except Exception:
                    res['bytetrack_ids'] = []

            self.csv_writer.writerow(res)

    def prepare_csv_reader_writer(self, source_path, video_prefix, csv_prefix=''):
        self.csv_reader_file = open(source_path / (video_prefix + csv_prefix + '.csv'), 'r')
        # self.csv_reader_file = open(source_path / (video_prefix + '_og.csv'), 'r')
        self.csv_reader = csv.reader(self.csv_reader_file, delimiter=',')
        fieldnames = self.csv_reader.__next__()  # Read headers

        self.csv_writer_file = open(source_path / (video_prefix + '_' + datetime.now().strftime("%m-%d-%Y:%H-%M-%S") + '.csv'), 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_writer_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def prepare_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.box_annotator = sv.BoxCornerAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)
        self.color_annotator = sv.ColorAnnotator(opacity=0.3, color=sv.ColorPalette(colors=[sv.Color(r=255, g=64, b=64), sv.Color(r=64, g=161, b=160)]))

    def check_box_inner(self, x, y):
        for id, box in enumerate(self.boxes_coordinates):
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return id, box
        return None, None

    def calc_selection_coords(self, ox, oy, x, y):
        sx, fx = (x, ox) if ox > x else (ox, x)
        sy, fy = (y, oy) if oy > y else (oy, y)

        sx = self.clip(self.cap_width, 0, sx)
        fx = self.clip(self.cap_width, 0, fx)
        sy = self.clip(self.cap_height, 0, sy)
        fy = self.clip(self.cap_height, 0, fy)

        return (sx, fx, sy, fy)

    def check_result_box_size(self, sx, fx, sy, fy):
        sx, fx = (sx - 4, fx + 4) if fx - sx < 2 else (sx, fx)
        sy, fy = (sy - 4, fy + 4) if fy - sy < 2 else (sy, fy)
        return (sx, fx, sy, fy)

    def lable_frame_state(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        mode_org = (30, 50)
        frame_org = (30, 100)
        fontScale = 1
        if self.event_handle_type == 0:
            mode_color = (0, 0, 255)
            mode_text = 'Del'
        elif self.event_handle_type == 1:
            mode_color = (0, 255, 0)
            mode_text = 'App'
        else:
            mode_color = (255, 255, 255)
            mode_text = '???'
            print('???')

        thickness = 2

        append_class_text = FLORENCE_REVERSE_ID_MAP[self.append_class] if self.florence else self.append_class
        frame = cv2.putText(
            frame, f'{mode_text}, {append_class_text}', mode_org, font,
            fontScale, mode_color, thickness, cv2.LINE_AA
        )

        frame = cv2.putText(
            frame, f'{self.frame_id}/{self.cap_frame_num-1}', frame_org, font,
            fontScale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        return frame

    def image_event(self, event, x, y, flags, params):
        if self.event_handle_type == 0:
            rect_selected = False
            if self.boxes_coordinates.size == 0:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                id, _ = self.check_box_inner(x, y)
                if id is not None:
                    self.select_rect = False

                    self.del_classes = np.append(self.del_classes, self.classes[id])
                    self.del_confidences = np.append(self.del_confidences, self.confidences[id])
                    if self.del_boxes_coordinates.size == 0:
                        self.del_boxes_coordinates = np.array([self.boxes_coordinates[id]])
                    else:
                        self.del_boxes_coordinates = np.append(self.del_boxes_coordinates, [self.boxes_coordinates[id]], axis=0)

                    self.classes = np.delete(self.classes, id)
                    self.confidences = np.delete(self.confidences, id)
                    self.boxes_coordinates = np.delete(self.boxes_coordinates, id, axis=0)

                    if self.florence:
                        self.del_bytetrack_ids = np.append(self.del_bytetrack_ids, self.bytetrack_ids[id])
                        self.bytetrack_ids = np.delete(self.bytetrack_ids, id)

                    self.annotated_frame = self.annotate_frame(self.raw_frame.copy())

                    frame = self.lable_frame_state(self.annotated_frame.copy())
                    cv2.imshow('frame', frame)

                # print(x, y, self.boxes_coordinates)
                print(self.del_classes, self.del_confidences, self.del_boxes_coordinates)

            elif event == cv2.EVENT_MOUSEMOVE:
                id, box = self.check_box_inner(x, y)
                if id is not None:
                    rect_selected = True
                    self.select_rect = True

                    box = box.astype(np.int32)
                    frame = self.annotated_frame.copy()
                    sub_img = frame[box[1]:box[3], box[0]:box[2]]
                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                    frame[box[1]:box[3], box[0]:box[2]] = res
                    frame = self.lable_frame_state(frame.copy())
                    cv2.imshow('frame', frame)

            if not rect_selected and self.select_rect:
                self.select_rect = False
                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)

        elif self.event_handle_type == 1:

            if event == cv2.EVENT_LBUTTONDOWN:
                self.left_mouse_state = True
                self.drawing_start = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.left_mouse_state == True:
                    frame = self.annotated_frame.copy()

                    sx, fx, sy, fy = self.calc_selection_coords(*self.drawing_start, x, y)
                    if sx - fx == 0 or sy - fy == 0:
                        return

                    sub_img = frame[sy:fy, sx:fx]
                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                    frame[sy:fy, sx:fx] = res

                    frame = self.lable_frame_state(frame.copy())
                    cv2.imshow('frame', frame)

            elif event == cv2.EVENT_LBUTTONUP:
                self.left_mouse_state = False

                sx, fx, sy, fy = self.calc_selection_coords(*self.drawing_start, x, y)
                sx, fx, sy, fy = self.check_result_box_size(sx, fx, sy, fy)

                self.classes = np.append(self.classes, self.append_class)
                self.confidences = np.append(self.confidences, 0.99)
                if self.boxes_coordinates.size == 0:
                    self.boxes_coordinates = np.array([[sx, sy, fx, fy]])
                else:
                    self.boxes_coordinates = np.append(self.boxes_coordinates, [[sx, sy, fx, fy]], axis=0)

                if self.florence:
                    self.bytetrack_ids = np.append(self.bytetrack_ids, 199)

                self.drawing_start = (None, None)

                self.annotated_frame = self.annotate_frame(self.raw_frame.copy())

                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)

    def annotate_frame(self, frame):
        if self.florence:
            class_names = [f'{FLORENCE_REVERSE_ID_MAP[cls]} #{bti}' for cls,bti in zip(self.classes, self.bytetrack_ids)]
        else:
            class_names = [self.classNames[cls] for cls in self.classes]

        # Create a sv.Detections object for annotation
        detections = sv.Detections(
            xyxy=self.boxes_coordinates,
            confidence=self.confidences,
            class_id=self.classes,
        )

        # Prepare labels for label annotator
        labels = [f"{class_name}" for class_name in class_names]

        # Annotate the image
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        if len(self.trace_coords) != 0:
            print(self.trace_coords, self.trace_classes)
            trace_detections = sv.Detections(
                xyxy=self.trace_coords,
                class_id=self.trace_classes
            )
            frame = self.color_annotator.annotate(scene=frame, detections=trace_detections)

        return frame

    def try_to_annotate(self):
        self.raw_frame = self.annotated_frame.copy()
        self.classes = np.array([], int)
        self.confidences = np.array([])
        self.boxes_coordinates = np.empty((0, 2))
        self.bytetrack_ids = np.array([])

        self.del_classes = np.array([], int)
        self.del_confidences = np.array([])
        self.del_boxes_coordinates = np.empty((0, 2))
        self.del_bytetrack_ids = np.array([])

        row = self.csv_reader.__next__()
        self.frame_id = row[0]

        class_data = json.loads(row[1])
        if class_data:
            confidences_data = json.loads(row[2])
            xyxy_data = json.loads(row[3])

            try:
                if self.florence:
                    bytetrack_ids = json.loads(row[4])
                    class_id, confidences, xyxy, by_ids = zip(
                        *((FLORENCE_ID_MAP[cls], conf, xy, bi) for cls, conf, xy, bi in zip(class_data, confidences_data, xyxy_data, bytetrack_ids) if
                            cls in FLORENCE_ID_MAP))
                    self.bytetrack_ids = np.array(by_ids, dtype=int)
                else:
                    class_id, confidences, xyxy = zip(
                        *((cls, conf, xy) for cls, conf, xy in zip(class_data, confidences_data, xyxy_data) if
                            cls in self.allowed_ids))

            except ValueError as exc:
                class_id, confidences, xyxy = [], [], []

            if class_id:
                self.classes = np.array(class_id, dtype=int)
                self.confidences = np.array(confidences, dtype=np.float32)
                self.boxes_coordinates = np.array(xyxy, dtype=np.float32)

                self.annotated_frame = self.annotate_frame(self.annotated_frame)

        frame = self.lable_frame_state(self.annotated_frame.copy())
        cv2.imshow('frame', frame)

    def revert_last_deleted(self):
        if self.del_classes.size != 0:
            self.classes = np.append(self.classes, self.del_classes[-1])
            self.confidences = np.append(self.confidences, self.del_confidences[-1])
            if self.boxes_coordinates.size == 0:
                self.boxes_coordinates = np.array([self.del_boxes_coordinates[-1]])
            else:
                self.boxes_coordinates = np.append(self.boxes_coordinates, [self.del_boxes_coordinates[-1]], axis=0)

            del_id = self.del_classes.size - 1

            self.del_classes = np.delete(self.del_classes, del_id)
            self.del_confidences = np.delete(self.del_confidences, del_id)
            self.del_boxes_coordinates = np.delete(self.del_boxes_coordinates, del_id, axis=0)

            if self.florence:
                self.bytetrack_ids = np.append(self.bytetrack_ids, self.del_bytetrack_ids[-1])
                self.del_bytetrack_ids = np.delete(self.del_bytetrack_ids, del_id)

            self.annotated_frame = self.annotate_frame(self.raw_frame.copy())

            frame = self.lable_frame_state(self.annotated_frame.copy())
            cv2.imshow('frame', frame)

    def process_video(self):
        # cap = cv2.VideoCapture(self.video_path)
        self.cap_frame_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.start_frame != 0:
            if self.start_frame > self.cap_frame_num - 1:
                print('The requested frame is further than the total length of the video')
                sys.exit(0)

            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame+1)
            for i in range(self.start_frame):
                _, _ = self.cap.read()
                row = self.csv_reader.__next__()
                res = {
                    'frame_id': row[0],
                    'classes': row[1],
                    'confidences': row[2],
                    'box_coords': row[3],
                }
                if self.florence:
                    res['bytetrack_ids'] = row[4]

                self.csv_writer.writerow(res)

        ret, self.annotated_frame = self.cap.read()

        self.raw_frame = self.annotated_frame.copy()

        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.try_to_annotate()

        frame = self.lable_frame_state(self.annotated_frame.copy())
        cv2.imshow('frame', frame)

        cv2.setMouseCallback('frame', self.image_event)

        while True:
            key = cv2.waitKey(0)
            self.left_mouse_state = False
            if key == ord('q'):  # Exit on 'q'
                if self.end_csv:
                    self.finalize_csv()
                self.cleanup()
                break
            elif key == ord('d'):
                self.event_handle_type = 0
                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)
                continue
            elif key == ord('a'):
                print('a press')
                self.event_handle_type = 1
                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)
                continue
            elif key == ord('1'):
                print('1 press')
                if self.florence:
                    self.append_class = FLORENCE_ID_MAP['inc']
                else:
                    self.append_class = 1
                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)
                continue
            elif key == ord('2'):
                print('2 press')
                if self.florence:
                    self.append_class = FLORENCE_ID_MAP['act']
                else:
                    self.append_class = 1
                frame = self.lable_frame_state(self.annotated_frame.copy())
                cv2.imshow('frame', frame)
                continue
            elif key == ord('z'):
                self.revert_last_deleted()
                continue
            elif key == ord('p'):  # Save image
                cv2.imwrite(
                    self.source_path / f'{self.video_path.stem}.jpg',
                    self.annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                continue
            elif key == 32 or key == ord('n'):  # space or n button
                res = {
                    'frame_id': self.frame_id,
                    'classes': json.dumps(self.classes.tolist() if not self.florence else [FLORENCE_REVERSE_ID_MAP[cls] for cls in self.classes]),
                    'confidences': json.dumps(self.confidences.tolist()),
                    'box_coords': json.dumps(self.boxes_coordinates.tolist()),
                }

                if self.florence:
                    res['bytetrack_ids'] = json.dumps(self.bytetrack_ids.tolist())

                if self.trace_ids:
                    self.trace_coords = []
                    self.trace_classes = []
                    for box, class_id in zip(self.boxes_coordinates, self.classes):
                        if class_id in self.trace_ids:
                            self.trace_coords.append(box)
                            self.trace_classes.append(class_id)
                    self.trace_coords = np.array(self.trace_coords)
                    self.trace_classes = np.array(self.trace_classes)

                self.csv_writer.writerow(res)
            else:
                continue

            ret, self.annotated_frame = self.cap.read()

            if not ret:
                print('End of video')
                self.cleanup()
                break

            self.try_to_annotate()


if __name__ == "__main__":
    video_name = Path('demo_3.mp4')
    source_path = Path('../data/videos/')
    # source_path = Path('/app/video_streaming_demo_1/videos/')

    dda = DispatchDemoAnnotator(video_name,
                                source_path,
                                csv_data_prefix='last',
                                start_frame=1050,
                                trace_ids=[199, 200])

    try:
        dda.process_video()
    except Exception as e:
        print('Exception during video showing: ', e)
        traceback.print_exc()

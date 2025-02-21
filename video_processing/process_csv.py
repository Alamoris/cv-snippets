import cv2
import csv
import json
import traceback
import logging
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO)


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
    source_path = Path('../data/videos/')
    video_file = source_path / video_name

    csv_reader_file, csv_reader = prepare_csv_reader(source_path / (video_name.stem + '.csv'))

    try:
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

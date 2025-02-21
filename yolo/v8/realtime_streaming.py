import cv2
import math
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GstRtspServer, GObject
from ultralytics import YOLO
import supervision as sv


loop = GObject.MainLoop()
GObject.threads_init()
Gst.init(None)


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        # Model
        model_path = "../WALDO-master/WALDO30_yolov8n_640x640.pt"
        self.model = YOLO(model_path)

        # Object classes
        self.classNames = [
            'LightVehicle',
            'Person',
            'Building',
            'UPole',
            'Container',
            'Boat',
            'Bike',
            'Container',
            'Truck',
            'Gastank',
            'Digger',
            'Solarpanels',
            'Bus'
        ]
        self.allowed_classes = ['LightVehicle', 'Person', 'Boat']

        # Annotators
        self.box_annotator = sv.BoxCornerAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2)

    def get_frame(self):
        ret, frame = self.cap.read()
        print('get frame', frame.shape)
        if not ret:
            print(f"End of video")
            raise Exception("End of video")
        frame = self.process_frame(frame)
        return frame

    def process_frame(self, frame):
        results = self.model(frame, stream=True)

        # Initialize lists to hold data for detections
        xyxy = []
        confidences = []
        class_ids = []
        class_names = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                if 0 <= cls < len(self.classNames) and self.classNames[cls] in self.allowed_classes:
                    class_name = self.classNames[cls]
                    class_names.append(class_name)

                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                    xyxy.append([x1, y1, x2, y2])
                    confidences.append(confidence)
                    class_ids.append(cls)

        # Check if there are any detections
        if xyxy:
            # Convert lists to numpy arrays
            xyxy = np.array(xyxy, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=int)

            # Create a sv.Detections object for annotation
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidences,
                class_id=class_ids
            )

            # Prepare labels for label annotator
            labels = [f"{class_name}" for class_name in class_names]

            # Annotate the image
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, video_processor, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.video_processor = video_processor
        self.number_frames = 0
        self.fps = 20
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=2560,height=1440,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)

    def on_need_data(self, src, lenght):
        try:
            frame = self.video_processor.get_frame()
            data = frame.tostring()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                                                                                   self.duration,
                                                                                   self.duration / Gst.SECOND))
            if retval != Gst.FlowReturn.OK:
                print(retval)
        except Exception as e:
            raise Exception('Something happend with the stream', e)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, video_processor, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(video_processor)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)


def start_rtsp_stream(video_path):
    #s = GstServer()
    vp = VideoProcessor(video_path)
    server = GstServer(vp)
    loop.run()


def cv2_test(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    video_path = 'videos/Rec_0003.mp4'
    start_rtsp_stream(video_path)


# https://mmpose.readthedocs.io/en/latest/user_guides/inference.html

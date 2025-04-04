import time
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


cap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.168.25:8554/main.264 latency=0 ! rtph265depay ! avdec_h265 ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER)

loop = GObject.MainLoop()
GObject.threads_init()
Gst.init(None)


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, video_path, **properties):
        super(SensorFactory, self).__init__(**properties)
        # self.cap = cv2.VideoCapture(video_path)
        self.cap = cap

        cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cat_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('cap_width, cat_height', cap_width, cat_height)
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             f'caps=video/x-raw,format=BGR,width={cap_width},height={cat_height},framerate={self.fps}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=medium tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'

    def on_need_data(self, src, lenght):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
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

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, video_path, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(video_path)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)


def start_rtsp_stream(video_path):
    #s = GstServer()
    server = GstServer(video_path)
    loop.run()


def cv2_test(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    video_path = '/videos/office_camera_test.mp4'
    # cap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.168.25:8554/main.264 latency=0 ! rtph265depay ! avdec_h265 ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER)
    time.sleep(2)
    ret, frame = cap.read()
    print(frame, "FRAME\n\n\n\n")

    start_rtsp_stream(video_path)
    #cv2_test(video_path)



"""
gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! rtph264depay ! avdec_h264 ! autovideosink sync=false
"""
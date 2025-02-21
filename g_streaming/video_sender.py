import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

loop = GObject.MainLoop()
GObject.threads_init()
Gst.init(None)


# class MyFactory(GstRtspServer.RTSPMediaFactory):
#     def __init__(self):
#         GstRtspServer.RTSPMediaFactory.__init__(self)

#     def do_create_element(self, url):
#         # s_src = "v4l2src ! video/x-raw,rate=30,width=320,height=240 ! videoconvert ! video/x-raw,format=I420"
#         # s_h264 = "videoconvert ! vaapiencode_h264 bitrate=1000"
#         s_src = "videotestsrc ! video/x-raw,rate=30,width=320,height=240,format=I420"
#         s_h264 = "x264enc tune=zerolatency"
#         pipeline_str = "( {s_src} ! queue max-size-buffers=1 name=q_enc ! {s_h264} ! rtph264pay name=pay0 pt=96 )".format(**locals())
#         if len(sys.argv) > 1:
#             pipeline_str = " ".join(sys.argv[1:])
#         print(pipeline_str)
#         return Gst.parse_launch(pipeline_str)


# class GstServer():
#     def __init__(self):
#         self.server = GstRtspServer.RTSPServer()
#         self.server.props.service = "%d" % 8554
#         f = MyFactory()
#         f.set_shared(True)
#         m = self.server.get_mount_points()
#         m.add_factory("/test", f)
#         self.server.attach(None)


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, video_path, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(video_path)
        self.number_frames = 0
        self.fps = 10
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=640,height=360,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)

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
    #video_path = '/videos/endurance-2024-12-04_21.22.24.mp4'
    video_path = 'videos/Rec_0003.mp4'
    start_rtsp_stream(video_path)
    #cv2_test(video_path)

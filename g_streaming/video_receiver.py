import cv2
import time
# vcap = cv2.VideoCapture("rtsp://192.168.0.39:8559/live")
vcap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.0.39:8559/live latency=300 ! rtph265depay ! avdec_h265 ! videoconvert ! video/x-raw ! appsink drop=1", cv2.CAP_GSTREAMER)
time.sleep(2)


while vcap.isOpened():
    ret, frame = vcap.read()
    print(ret, frame)
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

import cv2
import time
# vcap = cv2.VideoCapture("rtsp://192.168.168.25:8554/main.264")
# vcap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.144.25:8559/live latency=0 ! rtph265depay ! avdec_h265 ! videoconvert ! video/x-raw ! appsink drop=1", cv2.CAP_GSTREAMER)
vcap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.168.25:8554/main.264 latency=0 ! rtph265depay ! avdec_h265 ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER)
time.sleep(2)


while vcap.isOpened():
    ret, frame = vcap.read()
    print(ret, frame)
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

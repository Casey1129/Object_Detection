import struct

import cv2
import serial
import time
import numpy as np

ser = serial.Serial ("/dev/ttyAMA1", 115200)

threshold = 0.5  # Threshold to detect object

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObject(img, draw=True, objects = []):
    classIds, confidence, boundingBox = net.detect(img, confThreshold=threshold)
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            className = classNames[classId - 1]
            if className in objects:
                x = (box[0]+box[2])/2
                y = (box[1]+box[3])/2
                print(x, y)
                ser.write(str.encode('*'))
                ser.write(str.encode('*'))
                ser.write(struct.pack('ff', x, y))

                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)

    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    value1 = float(0)
    value2 = float(0)
    while True:
        success, img = cap.read()
        ser.write(str.encode('!'))
        ser.write(str.encode('!'))
        time.sleep(0.03)
        s = ser.read(100)
        (n1, n2) = struct.unpack('ff', s)
        objectIdentified = getObject(img, objects=['person'])
        cv2.imshow("Output", img)
        cv2.waitKey(1)


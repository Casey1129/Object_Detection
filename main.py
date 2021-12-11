import cv2

threshold = 0.5  # Threshold to detect object
#
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
# # cap.set(10,70) #Brightness of the camera

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
#
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
#
while True:
    success, img = cap.read()
    classIds, confidence, boundingBox = net.detect(img, confThreshold=threshold)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidence.flatten(), boundingBox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)

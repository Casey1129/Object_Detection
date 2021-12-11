from ObjectDetectionForPi import *

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
#cap.set(10,70) #Brightness of the camera
while True:
    success, img = cap.read()
    objectIdentified, objectInfo = getObject(img)
    cv2.imshow("Output", img)
    cv2.waitKey(1)

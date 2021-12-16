from ObjectDetectionForPi import *

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 320)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap.set(10,70) #Brightness of the camera
while True:
    success, img = cap.read()
    ser.write(str.encode('!'))
    ser.write(str.encode('!'))
    received_data = ser.read()
    time.sleep(0.03)
    data_left = ser.inWaiting()
    received_data += ser.read(data_left)
    (n1, n2) = struct.unpack('ff', received_data)
    print(n1)
    if np.isclose(n1, 1):
        x = 0
        y = 0
        objectIdentified = getObject(img, objects=['bottle'])
        cv2.imshow("Bottle", img)
        cv2.waitKey(1)

    if np.isclose(n1, 2):
        x = 0
        y = 0
        objectIdentified = getObject(img, objects=['bowl'])
        cv2.imshow("Bowl", img)
        cv2.waitKey(1)
    if np.isclose(n1, 3):
        x = 0
        y = 0
        objectIdentified = getObject(img, objects=['book'])
        cv2.imshow("Book", img)
        cv2.waitKey(1)

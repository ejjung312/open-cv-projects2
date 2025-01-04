import cv2
import math
import numpy as np
import util

from hand_tracking.HandTrackingModule import HandDetector


# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionConf=0.8, maxHands=1)

# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2) # deg(2차원)에 맞는 ax^2+bx+c a,b,c를 반환

while True:
    success, image = cap.read()
    hands, image = detector.findHands(image)

    if hands:
        lmList = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']
        x1, y1, _ = lmList[5] # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=ko#models
        x2, y2, _ = lmList[17]

        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        A, B, C = coff
        distanceCm = A*distance**2 + B*distance + C

        util.putTextRect(image, f'{int(distanceCm)} cm', (x, y))

    cv2.imshow("Image", image)
    cv2.waitKey(1)
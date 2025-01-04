import cv2
import numpy as np
import time

import hand_tracking.HandTrackingModule as htm
import pyautogui as gui

####################################
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
####################################

pTime = 0
prevLocX, prevLocY = 0, 0
curLocX, curLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam) # width
cap.set(4, hCam) # height

detector = htm.HandDetector(maxHands=1)
screen_width, screen_height = gui.size()

while True:
    # Find hand Landamarks
    success, image = cap.read()
    hands, image = detector.findHands(image)

    if hands:
        lmList, bbox = hands[0]['lmList'], hands[0]['bbox']

        # Get the tip of the index and middle finggers
        x1, y1, _ = lmList[8] # 검지
        x2, y2, _ = lmList[12] # 중지

        # Check which fingers are up
        fingers = detector.fingersUp(hands[0])

        cv2.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Only Index Finger(두번째 손가락): Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,screen_width)) # x1을 (0,wCam) 범위에서 (0,screen_width) 범위로 매핑
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,screen_height))

            # Smoothen Values
            # 평활화: 기존 위치 값과 새로운 위치 값을 부드럽게 변화시킴
            curLocX = prevLocX + (x3 - prevLocX) / smoothening
            curLocY = prevLocY + (y3 - prevLocY) / smoothening

            # Move Mouse
            gui.moveTo(screen_width-curLocX, curLocY)
            cv2.circle(image, (x1,y1), 15, (255,0,255), cv2.FILLED)

            prevLocX, prevLocY = curLocX, curLocY

        # Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, line, _ = detector.findDistance(lmList[8][1:], lmList[12][1:], image)

            # Click mouse if distance short
            if length < 30:
                cv2.circle(image, (line[4],line[5]), 15, (0,255,0), cv2.FILLED)
                gui.click()

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    # Display
    cv2.imshow("Image", image)
    cv2.waitKey(1)
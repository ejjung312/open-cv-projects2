import cv2
import numpy as np
import time
from motion_capture.PoseModule import PoseDetector

cap = cv2.VideoCapture("personal_ai_trainer/data/2.mp4")
detector = PoseDetector()
count = 0
direction = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("personal_ai_trainer/data/1.JPG")
    img = detector.findPose(img, False)
    lmList, bboxInfo = detector.findPosition(img, False)
    
    if len(lmList) != 0:
        # right arm
        # detector.findAngle(lmList[12], lmList[14], lmList[16], img)
        
        # left arm
        angle, img = detector.findAngle(lmList[11], lmList[13], lmList[15], img, scale=10)
        per = np.interp(angle, (210,310), (0,100))
        bar = np.interp(angle, (220,310), (650,100))
        
        color = (0,255,0)
        if per == 100:
            color = (0,0,255)
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 0:
            color = (0,255,0)
            if direction == 1:
                count += 0.5
                direction = 0
        
        # 막대 그리기
        cv2.rectangle(img, (1100,100), (1175,650), color, 3)
        cv2.rectangle(img, (1100,int(bar)), (1175,650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100,75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
        
        # 컬 카운트
        cv2.rectangle(img, (0,450), (250,720), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45,670), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0), 25)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
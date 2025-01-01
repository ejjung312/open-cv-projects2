import cv2
import numpy as np

from hand_tracking.HandTrackingModule import HandDetector
from util import cornerRect

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionConf=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200

class DragRect():
    def __init__(self, posCenter, size=[200,200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # x,y가 100~300사이(사각형 위치)에 있을 경우
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rectList = []
for x in range(5):
    rectList.append(DragRect([x*250+150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        cursor8 = lmList[8][1:] # 위치 정보 삭제
        cursor12 = lmList[12][1:]
        l, _, _ = detector.findDistance(cursor8, cursor12, img)

        if l<65: # 손가락 모았을 때 클릭했다고 간주
            for rect in rectList:
                rect.update(cursor8)

    # transparent and draw
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
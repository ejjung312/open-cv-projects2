import os
import cv2
import numpy as np

from motion_capture.PoseModule import PoseDetector
import util

cap = cv2.VideoCapture('shirt_try_on/data/1.mp4')
detector = PoseDetector()

shirtFolderPath = "shirt_try_on/shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262/190 # 셔츠너비 / 어깨11,12너비
shirtRatioHeightWidth = 581/440
imageNumber = 0
imgButtonRight = cv2.imread("shirt_try_on/data/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

while True:
    success, img = cap.read()
    
    img = detector.findPose(img)
    # img = cv2.flip(img, 1)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    
    if bboxInfo:
        # center = bboxInfo["center"]
        # cv2.circle(img, center, 5, (255,0,255), cv2.FILLED)
        lm11 = lmList[11][:-1]
        lm12 = lmList[12][:-1]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        
        widthOfShirt = int((lm11[0]-lm12[0])*fixedRatio)
        # print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt*shirtRatioHeightWidth)))
        currentScale = (lm11[0]-lm12[0])/190
        offset = int(44*currentScale), int(48*currentScale)
        
        
        try:
            img = util.overlayPNG(img, imgShirt, (lm12[0]-offset[0],lm12[1]-offset[1]))
        except:
            pass
        
        img = util.overlayPNG(img, imgButtonRight, (1074,293))
        img = util.overlayPNG(img, imgButtonLeft, (72,293))
        
        if lmList[16][0] < 300:
            counterRight += 1
            cv2.ellipse(img, (139,360), (66,66), 0, 0, counterRight*selectionSpeed, (0,255,0), 20)
            
            if counterRight*selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts)-1:
                    imageNumber += 1
        elif lmList[15][0] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138,360), (66,66), 0, 0, counterLeft*selectionSpeed, (0,255,0), 20)
            
            if counterLeft*selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
        else:
            counterRight = 0
            counterLeft = 0
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
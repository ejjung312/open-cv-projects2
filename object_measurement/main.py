import os
import cv2
import numpy as np

import object_measurement.util as util

webCam = False
path = os.path.join(os.getcwd(), "object_measurement", "2.jpg")
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

scale = 3
paper_width, paper_height = 210 * scale, 297 * scale

while True:
    if webCam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
        
    img, conts = util.getContours(img, minArea=50000, filter=4, draw=False)
    
    if len(conts) != 0:
        """
        4x1x2
        [
            [[413 161]] (1x2)
            [[ 84 166]]
            [[ 43 645]]
            [[451 646]]
        ]
        """
        biggest = conts[0][2] # 면적이 가장 큰 다각형 정보 (approx 값)
        imgWarp = util.warpImg(img, biggest, paper_width, paper_height)
        
        img2, conts2 = util.getContours(imgWarp, minArea=2000, filter=4, cThr=[50,50], draw=False)
        
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(img2, [obj[2]], True, (0,255,0), 2)
                nPoints = util.reorder(obj[2]) # 다각형 정보
                object_width = round(util.findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10, 1) # cm로 변환
                object_height = round(util.findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10, 1) # cm로 변환
                
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                
                x, y, w, h = obj[3]
                cv2.putText(img2, '{}cm'.format(object_width), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255), 2)
                cv2.putText(img2, '{}cm'.format(object_height), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255), 2)
                
        cv2.imshow('A4', img2)
    
    # img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow('original', img)
    
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    break
    
# 자원 해제
cap.release()
cv2.destroyAllWindows()
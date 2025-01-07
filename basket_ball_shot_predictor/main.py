import cv2
import numpy as np
import math

from basket_ball_shot_predictor.ColorModule import ColorFinder
import util as util


# 초기화
cap = cv2.VideoCapture('basket_ball_shot_predictor/Videos/vid (4).mp4')

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

posListX, posListY = [], []
xList = [item for item in range(0,1300)]
prediction = False

while True:
    success, img = cap.read()
    # img = cv2.imread("basket_ball_shot_predictor/Ball.png")
    img = img[0:900, :] # 높이만 자름
    
    # 공 찾기
    imgColor, mask = myColorFinder.update(img, hsvVals)
    
    # 공 위치 찾기
    imgContours, contours = util.findContours(img, mask ,minArea=500)
    
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
        
    if posListX:
        # 다항 회귀 y = Ax^2 + Bx + C
        # deg(2차원)에 맞는 ax^2+bx+c a,b,c를 반환
        A, B, C = np.polyfit(posListX, posListY, 2)
        
        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            
            cv2.circle(imgContours, pos, 10, (0,255,0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0,255,0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i-1],posListY[i-1]), (0,255,0), 2) # 이전 좌표까지 선 긋기
        
        for x in xList:
            y = int(A*x**2 + B*x + C)
            cv2.circle(imgContours, (x,y), 2, (255,0,255), cv2.FILLED)
    
        # 예측
        # X: 330~430, Y: 590 일 때 성공으로 예측
        if len(posListX) < 10:
            a = A
            b = B
            c = C-590
            x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a)) # https://blog.naver.com/since201109/220728976399?photoView=0
            
            prediction = 330 < x < 430
            
        if prediction:
            print('Basket')
            util.putTextRect(imgContours, "Basket", (50,150), scale=5, thickness=5, colorR=(0,200,0), offset=20)
        else:
            print('No Basket')
            util.putTextRect(imgContours, "No Basket", (50,150), scale=5, thickness=5, colorR=(0,0,200), offset=20)
        
    
    # display
    imgContours = cv2.resize(imgContours, (0,0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("imgColor", imgContours)
    cv2.waitKey(100)
import cv2
import numpy as np

def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    
    if showCanny:
        cv2.imshow('Canny', imgThre)
    
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finalCountours = []
    # N x [[x1, y1]]
    for i in contours: # (N,1,2) N:점의 총 수, 1: 하나의 좌표 쌍, 2: x,y 좌표값
        area = cv2.contourArea(i) # 윤곽선 면적 계산
        if area > minArea:
            peri = cv2.arcLength(i, True) # 윤곽선 둘레 길이 계산 
            approx = cv2.approxPolyDP(i, 0.02*peri, True) # 윤곽선을 다각형으로 근사화
            bbox = cv2.boundingRect(approx)
            
            if filter > 0:
                if len(approx) == filter:
                    # 꼭지점개수, 면적, 꼭지점정보, 바운딩박스, xy값
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])

    # x에서 x[1]을 추출하여 이를 기준으로 정렬 => 면적 크기로 오름차순
    finalCountours = sorted(finalCountours, key=lambda x:x[1], reverse=True)
    
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0,0,255), 3)
    
    return img, finalCountours


def reorder(myPoints):
    """
    4x2
    [
        [413 161]
        [ 84 166]
        [ 43 645]
        [451 646]
    ]
    """
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1) # x,y좌표를 더함
    
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    print(myPointsNew)
    print(myPointsNew.shape)
    
    return myPointsNew


def warpImg(img, points, w, h, pad=20):
    points = reorder(points)
    
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w, 0], [0, h], [w, h]])
    
    # 설정한 w, h로 맞춤
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
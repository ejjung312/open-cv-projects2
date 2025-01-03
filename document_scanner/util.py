import cv2
import numpy as np

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # 네 꼭지점 좌표를 4x2 형태로 재구성
    myPointsNew = np.zeros((4,1,2), dtype=np.int32)
    """
    [[[0 0]]
    [[0 0]]
    [[0 0]]
    [[0 0]]]
    """
    add = myPoints.sum(1) # x,y 좌표를 더함
    
    myPointsNew[0] = myPoints[np.argmin(add)] # 합이 가장 작은 좌표는 왼쪽 상단. np.argmin(add): 최소값 인덱스
    myPointsNew[3] = myPoints[np.argmax(add)] # 합이 가장 큰 좌표는 오른쪽 하단
    diff = np.diff(myPoints, axis=1) # 각 행의 차이 계산 (y-x)
    myPointsNew[1] = myPoints[np.argmin(diff)] # diff가 가장 작은 좌표는 오른쪽 상단
    myPointsNew[2] = myPoints[np.argmax(diff)] # diff가 가장 큰 좌표는 왼쪽 하단
    
    return myPointsNew
    

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    
    for i in contours:
        area = cv2.contourArea(i) # 윤곽선 면적 계산
        if area > 5000:
            peri = cv2.arcLength(i, True) # 윤곽선 둘레 길이 계산 (True: 폐곡선)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) # 운곽선을 다각형으로 근사화
            
            if area > max_area and len(approx) == 4: # 면적이 가장 크고 꼭지점이 4개인 경우
                biggest = approx # 가장 큰 사각형 업데이트
                max_area = area # 면적 업데이트
            
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    
    return img


def nothing(x):
    pass


def initializeTrackbars(values):
    def on_trackbar_change(_):
        values['threshold1'] = cv2.getTrackbarPos("Threshold1", "Trackbars")
        values['threshold2'] = cv2.getTrackbarPos("Threshold2", "Trackbars")
    
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", values['threshold1'], 255, on_trackbar_change)
    cv2.createTrackbar("Threshold2", "Trackbars", values['threshold2'], 255, on_trackbar_change)
import os
import cv2
import pickle
import numpy as np

from util import putTextRect

video_path = os.path.join(os.getcwd(), "parking_space_counter", "carPark.mp4")

pos_path = os.path.join(os.getcwd(), "parking_space_counter", "CarParkPos")

width, height = 107, 48

def checkParkingSpace(imgPro):
    spaceCounter = 0
    
    for pos in posList:
        x, y = pos
        
        imgCrop = imgPro[y:y+height, x:x+width]
        # 이미지 내의 0이 아닌 픽셀의 개수를 세는 함수.
        # 그레이 이미지로 변환한 뒤 이진화를 통해 넘어온 이미지를 통해서 계산함
        count = cv2.countNonZero(imgCrop)
        
        if count < 900:
            # 주차가능
            color = (0,255,0)
            thickness = 5
            spaceCounter += 1
        else:
            # 주차불가능
            color = (0,0,255)
            thickness = 2
        
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)
        putTextRect(img, str(count), (x, y+height-3), scale=1, thickness=2, offset=0, colorR=(0,0,255))
    
    putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=color)

# video feed
cap = cv2.VideoCapture(video_path)

with open(pos_path, 'rb') as f:
    posList = pickle.load(f)

while True:
    # 현재 읽고 있는 프레임의 인덱스 == 비디오의 총 프레임 수
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # 비디오의 프레임 포인터를 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러- 노이즈 제거, 이미지 부드럽게 하기(스무딩), 엣지 검출 전처리 등에 사용
    # 이미지, 커널사이즈, X축방향의 표준편차(값이 클수록 블러 효과가 강해짐)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    
    # 픽셀별로 임계값 계산
    # 이미지, 임계값 최대값, 임계값 계산방법(ADAPTIVE_THRESH_GAUSSIAN_C: 가우시간 가중 평균 기반), 이진화방법(임계값 이상인 픽셀은 0, 나머지는 최대값), 임계값 계산할 영역크기(홀수), 보정상수(계산된 임계값에서 빼는 값)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    
    # 중간값 필터링, 노이즈 제거에 사용
    # 입력이미지, 커널크기(홀수)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    
    kernel = np.ones((3,3), np.int8)
    # 팽창연산. 흰색 영역을 확장시켜 노이즈 제거, 객체 강조 등에 사용
    # 이미지, 커널, 반복횟수
    imDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    
    checkParkingSpace(imDilate)
    
    cv2.imshow("Image", img)
    # cv2.imshow("Image", imDilate)
    cv2.waitKey(10)
import os
import cv2
import cv2.aruco as aruco
import numpy as np

def loadArguImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total Number of Markers Detected:", noOfMarkers)
    argDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgArg = cv2.imread(f'{path}/{imgPath}')
        argDics[key] = imgArg
    
    return argDics


def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ArUco 딕셔너리 선택 (6x6 마커, 250개의 프리셋 마커)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    
    # ArUco 파라미터 설정
    arucoParam = aruco.DetectorParameters()
    
    # ArUco 감지기 생성 (OpenCV 4.x 버전)
    detector = aruco.ArucoDetector(arucoDict, arucoParam)
    
    # ArUco 마커 감지
    corners, ids, rejected = detector.detectMarkers(imgGray)
    
    if ids is not None and draw:
        aruco.drawDetectedMarkers(img, corners)

    return [corners, ids]


def argumentAruco(corner, id, img, imgAug, drawId=True):
    top_l = corner[0][0][0], corner[0][0][1]
    top_r = corner[0][1][0], corner[0][1][1]
    btm_r = corner[0][2][0], corner[0][2][1]
    btm_l = corner[0][3][0], corner[0][3][1]
    
    h, w, c = imgAug.shape
    
    pts1 = np.array([top_l, top_r, btm_r, btm_l])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    """
    cv2.findHomography: 다수의 특징점 매칭 후 Outlier 제거를 통해 호모그래피 행렬을 계산할 때 사용
    cv2.getPerspectiveTransform: 4개의 대응점을 사용하여 단순한 투시 변환 행렬을 계산할 때 사용
    """
    matrix, _ = cv2.findHomography(pts2, pts1) # 입력 점들 간의 변환 관계를 모델링하여 투시 변환을 수행
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1],img.shape[0])) # imgAug가 pts1 영역에 투시 변환된 이미지
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0)) # 이미지에서 볼록 다각형(Convex Polygon)을 채우는 데 사용. img의 pts1 영역을 (0,0,0)으로 채워서 덮음
    
    # imgOut = img + imgOut # 두 이미지의 픽셀 값을 더해 이미지를 합침
    imgOut = cv2.add(img, imgOut) # 오버플로우 예방
    
    if drawId:
        top_l = (int(top_l[0]), int(top_l[1]))
        cv2.putText(imgOut, str(id), top_l, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    
    return imgOut


def main():
    
    webCam = False
    cap = cv2.VideoCapture(0)
    
    imgAug = cv2.imread("aruco_markers/markers/23.jpg")
    argDics = loadArguImages("aruco_markers/markers")
    
    while True:
        if webCam:
            success, img = cap.read()
        else:
            # img = cv2.imread(os.path.join("aruco_markers", "marker.jpg"))
            # img = cv2.imread(os.path.join("aruco_markers", "marker1.jpg"))
            img = cv2.imread(os.path.join("aruco_markers", "marker2.jpg"))
        
        arucoFound = findArucoMarkers(img)
        
        if len(arucoFound[0]) != 0:
            for corner, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in argDics.keys():
                    img = argumentAruco(corner, id, img, argDics[int(id)])
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
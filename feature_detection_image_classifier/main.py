import os
import cv2
import numpy as np

webCam = False
path = "feature_detection_image_classifier/images_query"
orb = cv2.ORB_create(nfeatures=1000) # 특징점 검출 및 descriptor 추출 알고리즘
cap = cv2.VideoCapture(0)
img_path = os.path.join(os.getcwd(), "feature_detection_image_classifier/images_train", "2.jpg")

### Import Images
images = []
classNames = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])


def findDes(images):
    desList = []
    for img in images:
        keypoints, des = orb.detectAndCompute(img, None) # ORB 알고리즘을 사용하여 이미지에서 특징점과 기술자를 동시에 검출 및 계산. gray scale 이미지여야 정확한 결과를 얻을 수 있음
        desList.append(des)
    return desList


def findId(img, desList, thres=15):
    keypoints, des2 = orb.detectAndCompute(img, None)
    
    bf = cv2.BFMatcher() # 브루트 포스 방식을 사용해 두 이미지의 특징점 기술자를 비교
    matchList = []
    finalVal = -1
    try:
        for des1 in desList:
            matches = bf.knnMatch(des1, des2, k=2) # 최근접 이웃 알고리즘을 사용해 기술자에 대해 k개의 가장 가까운 매칭을 찾음
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
        # print(matchList) # 높은 수의 인덱스가 매칭됨을 뜻함
    except:
        pass
    
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))

    return finalVal


desList = findDes(images)
print(len(desList))

while True:    
    if webCam:
        success, img2 = cap.read()
    else:
        img2 = cv2.imread(img_path)
    
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    id = findId(img2, desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    cv2.imshow("img2", imgOriginal)
    cv2.waitKey(1)


# img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, flags=2)

# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
# cv2.imshow("img3", img3)
# # cv2.imshow('Kp1', imgKp1)
# # cv2.imshow('Kp2', imgKp2)
# cv2.waitKey(0)
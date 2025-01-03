import os

import cv2
import numpy as np
import pytesseract
import document_scanner.util as util

img_path = os.path.join(os.getcwd(), "document_scanner", "data", "1.jpg")

height, width = 1024, 768

values = {'threshold1': 100, 'threshold2': 200}

# util.initializeTrackbars(values)

# 샤프닝 커널
# sharpen_kernel = np.array([[0, -1, 0],
#                         [-1, 5, -1],
#                         [0, -1, 0]])
kernel = np.ones((5, 5))


img = cv2.imread(img_path)
img = cv2.resize(img, (width, height))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 이미지, 커널크기, X축 방향의 표준 편차(sigmaX)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)

imgThreshold = cv2.Canny(imgBlur, values['threshold1'], values['threshold2'])

imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # 팽창
imgErode = cv2.erode(imgDial, kernel, iterations=1) # 침식

# 외곽선 정보 검출
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

# 최대 외곽선 검출
biggest, maxArea = util.biggestContour(contours)

if biggest.size != 0:
    biggest = util.reorder(biggest)
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0,255,0), 20)
    imgBigContour = util.drawRectangle(imgBigContour, biggest, 2)
    
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # 설정한 width, height로 맞춤
    imgWrapColored = cv2.warpPerspective(img, matrix, (width, height))
    
    # 샤프닝 커널 필터
    # imgSharpened1 = cv2.filter2D(imgWrapColored, -1, sharpen_kernel)
    
    blurred = cv2.GaussianBlur(imgWrapColored, (5,5), 0)
    # 언샤프 마스크
    # imgSharpened2 = cv2.addWeighted(imgWrapColored, 1.5, blurred, -0.5, 0)
    
    # 고주파강조
    high_pass = cv2.subtract(imgWrapColored, blurred)
    imgSharpened3 = cv2.add(imgWrapColored, high_pass)
    
    
    imgWrapGray = cv2.cvtColor(imgSharpened3, cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWrapGray, 255, 1, 1, 7, 2) # 이진이미지(흑백이미지)로 변경
    # imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre) # 이미지 색상 반전
    imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3) # 노이즈 제거
    
    # ocr 실행
    # text = pytesseract.image_to_string(imgAdaptiveThre, lang='kor')
    # print(text)

    cv2.imshow("img", img)
    # cv2.imshow("imgGray", imgGray)
    # cv2.imshow("imgBlur", imgBlur)
    # cv2.imshow("imgThreshold", imgThreshold)
    # cv2.imshow("imgDial", imgDial)
    # cv2.imshow("imgErode", imgErode)
    # cv2.imshow("imgContours", imgContours)
    # cv2.imshow("imgBigContour", imgBigContour)
    # cv2.imshow("imgWrapColored", imgWrapColored)
    # cv2.imshow("imgWrapColored", imgWrapColored)
    # cv2.imshow("imgSharpened1", imgSharpened1)
    # cv2.imshow("imgSharpened2", imgSharpened2)
    cv2.imshow("imgSharpened3", imgSharpened3)
    # cv2.imshow("imgAdaptiveThre", imgAdaptiveThre)

    # 무한대기
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
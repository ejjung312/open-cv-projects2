import os
import cv2
import numpy as np
import util as util

##############################################
path = os.path.join(os.getcwd(), "omr_automated_grading", "data", "1.JPG")
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4]
webcameFeed = False
cameraNo = 0
##############################################

cap = cv2.VideoCapture(cameraNo)
cap.set(10, 150)

while True:
    if webcameFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # preprocessing
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    
    try:
        # finding all contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

        # find rectangles
        rectCon = util.rectContour(contours)
        biggestContour = util.getCornerPoints(rectCon[0])
        gradePoints = util.getCornerPoints(rectCon[1])

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)
            
            biggestContour = util.reorder(biggestContour)
            gradePoints = util.reorder(gradePoints)
            
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg,heightImg))
            
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0,0], [325,0], [0,150], [325,150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325,150))
            # cv2.imshow("grade", imgGradeDisplay)
            
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            
            # Getting no zero pixel values of each box
            boxes = util.splitBoxes(imgThresh)
            myPixelVal = np.zeros((questions,choices))
            countC = 0
            countR = 0
            for image in boxes:
                # 이미지 내의 0이 아닌 픽셀의 개수를 세는 함수.
                # 그레이 이미지로 변환한 뒤 이진화를 통해 넘어온 이미지를 통해서 계산함
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0
            
            # Finding index values of the markings 
            myIndex = [] # 선택한 값 리스트
            for x in range(0, questions):
                arr = myPixelVal[x]
                # np.amax(arr): 배열에 가장 큰 값
                myIndexVal = np.where(arr==np.amax(arr)) # 배열에서 가장 큰 값의 인덱스 반환
                myIndex.append(myIndexVal[0][0])
            
            # Grading
            grading = []
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            
            score = (sum(grading)/questions) * 100 # Final Grade
            
            # Display answers
            imgResult = imgWarpColored.copy()
            imgResult = util.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = util.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
            
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg,heightImg))
            
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score))+"%", (60,100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 3)
            cv2.imshow("Grade", imgRawGrade)
            
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGrade = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg,heightImg))
            
            # 이미지 합치기
            # alpha: 첫번째 이미지의 가중치. 값이 클 수록 첫번째 이미지가 강조됨
            # beta: 두번째 이미지의 가중치. 값이 클 수록 두번째 이미지가 강조됨
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGrade, 1, 0)
            
            
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny], 
                    [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                    [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
    
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny], 
                    [imgBlank, imgBlank, imgBlank, imgBlank],
                    [imgBlank, imgBlank, imgBlank, imgBlank])
    
    labels = [["Original", "Gray", "Blur", "Canny"], ["Contours", "BiggestContours", "WarpColored", "Thresh"], ["Result", "RawDrawing", "InvWarp", "Final"]]
    imgStacked = util.stackImages(imageArray, 0.3, labels)

    cv2.imshow("original", imgStacked)
    cv2.imshow("Final Result", imgFinal)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    
cv2.destroyAllWindows()
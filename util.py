import cv2
import numpy as np

def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list) #  객체가 특정 클래스나 데이터 타입의 인스턴스인지 확인
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)
                
                # 2차원 이미지(흑백 이미지)를 3차원 이미지(BGR)로 변환
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
                    
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x]) # 가로 방향으로 병합
            hor_con[x] = np.concatenate(imgArray[x]) # 가로 방향으로 병합
            
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1]/cols)
        eachImgHeight = int(ver.shape[0]/rows)
        
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(labels[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver, str(labels[d][c]),(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    
    return ver
    

def cornerRect(img, bbox, l=30, t=5, rt=1, colorR=(255,0,255), colorC=(0,255,0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x+w, y+h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)

    # Top Left x,y
    cv2.line(img, (x,y), (x+l,y), colorC, t)
    cv2.line(img, (x,y), (x,y+l), colorC, t)

    # Top Right x1,y
    cv2.line(img, (x1,y), (x1-l,y), colorC, t)
    cv2.line(img, (x1, y), (x1, y+l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x+l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1-l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1-l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1-l), colorC, t)

    return img

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255,255,255), colorR=(255,0,255), font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0,255,0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, thickness)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    
    return img, [x1, y1, x2, y2]


def rectContour(contours):
    rectCon = []
    
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area", area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            # print("Corner Points ", len(approx))
            if len(approx) == 4:
                rectCon.append(i)
    
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True)
    
    return approx


def reorder(myPoints):
    # (4,1,2) -> (4,2)
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    
    add = myPoints.sum(1) # x축의 값 더함 (1,4)
    myPointsNew[0] = myPoints[np.argmin(add)] # [0,0]
    myPointsNew[3] = myPoints[np.argmax(add)] # [width,height]
    diff = np.diff(myPoints, axis=1)
    
    myPointsNew[1] = myPoints[np.argmin(diff)] # [width,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] # [height,0]
    
    return myPointsNew


def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split", box)
            
    return boxes


def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    
    for x in range(0,questions):
        myAns = myIndex[x]
        cX = (myAns*secW) + secW//2
        cY = (x*secH) + secH//2

        if grading[x] == 1:
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH)+secH//2), 20, (0,255,0), cv2.FILLED)
            
        cv2.circle(img, (cX,cY), 50, myColor, cv2.FILLED)
        
    return img


def findContours(img, imgPre, minArea=1000, maxArea=float('inf'), sort=True,
                filter=None, drawCon=True, c=(255,0,0), ct=(255,0,255),
                retrType=cv2.RETR_EXTERNAL, approxType=cv2.CHAIN_APPROX_NONE):
    
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgPre, retrType, approxType)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if minArea < area < maxArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            
            if filter is None or len(approx) in filter:
                if drawCon:
                    cv2.drawContours(imgContours, cnt, -1, c, 3)
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.putText(imgContours, str(len(approx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ct, 2)
                cx, cy = x + (w // 2), y + (h // 2)
                cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED)
                conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})
                
    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True) 
    
    return imgContours, conFound


def overlayPNG(imgBack, imgFront, pos=[0,0]):
    hf,wf,cf = imgFront.shape
    hb,wb,cb = imgBack.shape
    
    x1,y1 = max(pos[0],0), max(pos[1],0) # 전경 이미지가 배경 이미지 위에 그려질 시작 좌표 (왼쪽 상단)
    x2,y2 = min(pos[0]+wf,wb), min(pos[1]+hf,hb) # 전경 이미지가 배경 이미지 위에 그려질 끝 좌표 (오른쪽 하단)
    
    x1_overlay = 0 if pos[0] >= 0 else -pos[0] # 전경 이미지의 유효한 x,y 시작점
    y1_overlay = 0 if pos[1] >= 0 else -pos[1]
    
    wf, hf = x2-x1, y2-y1 # 전경 이미지가 실제로 배경 이미지에 그려질 영역의 너비(wf)와 높이(hf)를 다시 계산
    
    if wf <= 0 or hf <= 0:
        return imgBack
    
    alpha = imgFront[y1_overlay:y1_overlay+hf, x1_overlay:x1_overlay+wf, 3]/255.0
    inv_alpha = 1.0-alpha
    
    imgRGB = imgFront[y1_overlay:y1_overlay+hf, x1_overlay:x1_overlay+wf, 0:3]
    
    for c in range(0, 3):
        # 블렌딩 공식: 최종 픽셀=(배경 픽셀×(1−α))+(전경 픽셀×α)
        imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c]*inv_alpha + imgRGB[:,:,c]*alpha
        
    return imgBack
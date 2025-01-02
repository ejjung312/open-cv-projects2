import os
import cv2
import pickle
import numpy as np

image_path = os.path.join(os.getcwd(), "parking_space_counter", "carParkImg.png")

width, height = 107, 48

try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1+width and y1 < y < y1+height:
                posList.pop(i)
                
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

while True:
    
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (255,0,255), 2)
    
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    cv2.waitKey(1)
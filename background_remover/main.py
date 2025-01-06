import os
import cv2
from background_remover.SelfieSegmentationModule import SelfieSegmentation
from background_remover.FPS import FPS
import util as util

video_path = os.path.join(os.getcwd(), "background_remover/video1.mp4")
bg_path = os.path.join(os.getcwd(), "background_remover/data/1.jpg")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfieSegmentation(model=0)
fpsReader = FPS()
imgBg = cv2.imread(bg_path)

listImg = os.listdir("background_remover/data")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'background_remover/data/{imgPath}')
    imgList.append(img)

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], cutThreshold=0.75)
    
    imgStacked = util.stackImages([img, imgOut], 0.5)
    _, imgStacked = fpsReader.update(imgStacked, bgColor=(0,0,255))
    
    cv2.imshow("Image", imgStacked)
    
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
        else:
            indexImg = 0
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1
        else:
            indexImg = len(imgList)-1
    elif key == ord('q'):
        break
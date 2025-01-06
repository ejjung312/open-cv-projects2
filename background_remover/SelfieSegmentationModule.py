import cv2
import mediapipe as mp
import numpy as np

import util as util

class SelfieSegmentation():
    def __init__(self, model=1):
        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(model_selection=self.model)
        
    def removeBG(self, img, imgBg=(255,255,255), cutThreshold=0.1):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 세그멘테이션 마스크 생성. 객체 영역을 나타내는 2D 세그멘테이션 마스크
        results = self.selfieSegmentation.process(imgRGB)
        # 2D 세그멘테이션 마스크 -> 3채널 RGB 이미지로 
        # (results.segmentation_mask, ) 튜플 값을 3번 반복(*3)
        # cutThreshold를 초과하면 True(객체), 아니면 False(배경)
        condition = np.stack((results.segmentation_mask, )*3, axis=-1) > cutThreshold
        
        if isinstance(imgBg, tuple):
            # img와 동일한 크기의 검정색 이미지 생성
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            # 배열의 모든 픽셀을 imgBG 값으로 채움
            _imgBg[:] = imgBg
            # condition == True: img 픽셀 사용
            # condition == False: _imgBg 픽셀 사용
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, imgBg)
        
        return imgOut
    

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    # model is 0 or 1 - 0 is general 1 is landscape(faster)
    segmentor = SelfieSegmentation(model=0)
    
    while True:
        success, img = cap.read()
        
        imgOut = segmentor.removeBG(img, imgBg=(255,0,255), cutThreshold=0.1)
        
        imgStacked = util.stackImages([img, imgOut], cols=2, scale=1)
        
        cv2.imshow("Image", imgStacked)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
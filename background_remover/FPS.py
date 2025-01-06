import time
import cv2
import util as util

"""
FPS class for calculating and displaying the Frames Per Second in a video stream.
"""
class FPS:
    def __init__(self, avgCount=30):
        self.pTime = time.time()
        self.frameTimes = []
        self.avgCount = avgCount
        
    def update(self, img=None, pos=(20,50), bgColor=(255,0,255), 
                textColor=(255,255,255), scale=3, thickness=3):
        """
        Update the frame rate and optionally display it on the image.
        """
        cTime = time.time()
        # 이전프레임과 현재프레임 시간차 계산
        frameTime = cTime - self.pTime 
        self.frameTimes.append(frameTime)
        self.pTime = cTime
        
        if len(self.frameTimes) > self.avgCount:
            self.frameTimes.pop(0)
        
        # 평균 프레임 계산
        avgFrameTime = sum(self.frameTimes) / len(self.frameTimes)
        fps = 1 / avgFrameTime # fps 계산
        
        if img is not None:
            util.putTextRect(img, f'FPS: {int(fps)}', pos, 
                            scale=scale, thickness=thickness, 
                            colorT=textColor, colorR=bgColor, offset=10)
            
        return fps, img
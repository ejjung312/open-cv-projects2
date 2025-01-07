import cv2
import numpy as np

class ColorFinder:
    def __init__(self, trackBar=False):
        self.trackBar = trackBar
        if self.trackBar:
            self.initTrackbars()
    
    def empty(self, a):
        pass
    
    def initTrackbars(self):
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, self.empty)
        cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, self.empty)
        cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, self.empty)
        cv2.createTrackbar("Val Min", "TrackBars", 0, 255, self.empty)
        cv2.createTrackbar("Val Max", "TrackBars", 255, 255, self.empty)
        
    def getTrackbarValues(self):
        hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
        smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
        vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
        hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
        smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
        vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
        
        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                "hmax": hmax, "smax": smax, "vmax": vmax}

        return hsvVals
    
    def update(self, img, myColor=None):
        imgColor = []
        mask = []
        
        if self.trackBar:
            myColor = self.getTrackbarValues()
        
        if isinstance(myColor, str):
            myColor = self.getColorHSV(myColor)
        
        if myColor is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([myColor['hmin'], myColor['smin'], myColor['vmin']])
            upper = np.array([myColor['hmax'], myColor['smax'], myColor['vmax']])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgColor = cv2.bitwise_and(img, img, mask=mask) # mask에서 0이 아닌 픽셀에 해당하는 영역만 AND 연산이 적용되어 이미지가 보존 됨. 나머지는 0으로 설정
        
        return imgColor, mask
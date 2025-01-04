import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, image = cap.read()

    code = decode(image)
    for barcode in decode(image):
        myData = barcode.data.decode('utf-8')
        pts = np.array([barcode.polygon], np.int32) # 1x4x2 텐서
        pts = pts.reshape((-1,1,2)) # 4x1x2
        cv2.polylines(image, [pts], True, (255,0,255), 5)

        pts2 = barcode.rect
        cv2.putText(image, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

    cv2.imshow('Result', image)
    cv2.waitKey(1)
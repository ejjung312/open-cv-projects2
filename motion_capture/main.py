import cv2
from motion_capture.PoseModule import PoseDetector

cap = cv2.VideoCapture('motion_capture/data/Video.mp4')

detector = PoseDetector()
posList = []

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
        posList.append(lmString)

    print(len(posList))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open("motion_capture/data/AnimationFile.txt", 'w') as f:
            f.writelines(["%s\n" % item for item in posList])
    
    """
    생성된 AnimationFile.txt은 유니티에서 모션 캡쳐로 사용 가능
    참고영상: https://www.youtube.com/watch?v=BtMs0ysTdkM
    """
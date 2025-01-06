import os
import cv2

# img_path = os.path.join(os.getcwd(), "object_detection_mobilenet", "lena.png")
# img = cv2.imread(img_path)

video_path = os.path.join(os.getcwd(), "object_detection_mobilenet", "data1.mp4")
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)
cap.set(4, 480)


classNames = []
classFile = os.path.join(os.getcwd(), "object_detection_mobilenet", "coco.names")

# rt: 텍스트 모드로 읽기 전용
with open(classFile, 'rt') as f:
    # rstrip('\n'): 문자열 끝에 있는 줄 바꿈 문자 제거
    # split('\n'): 문자열 줄 바꿈 문자를 기준으로 나눔
    classNames = f.read().rstrip('\n').split('\n')

configPath = os.path.join(os.getcwd(), "object_detection_mobilenet", 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(os.getcwd(), "object_detection_mobilenet", 'frozen_inference_graph.pb')

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320) # 입력 크기
net.setInputScale(1.0/127.5) # 스케일링 비율 설정 (정규화). [-1,1] 범위로 정규화
net.setInputMean((127.5, 127.5, 127.5)) # 이미지의 평균값 설정. RGB 채널에서 평균값을 뺌
net.setInputSwapRB(True) # BGR을 RGB로 변경

while True:
    success, img = cap.read()
    
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=3)
            cv2.putText(img, classNames[classIds-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow("output", img)
    cv2.waitKey(1)
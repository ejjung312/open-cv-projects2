import os
import cv2
import numpy as np
import face_recognition

path = 'data2'
images = []
class_names = []
my_list = os.listdir(path)

for cls in my_list:
    current_image = cv2.imread(f'{path}/{cls}')
    images.append(current_image)
    class_names.append(os.path.splitext(cls)[0])


def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


encode_list_known = findEncodings(images)
print('Encoding Complete')


cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    imgS = cv2.resize(image, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, face_locations)

    for encode, location in zip(encodes, face_locations):
        matches = face_recognition.compare_faces(encode_list_known, encode)
        face_distance = face_recognition.face_distance(encode_list_known, encode)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_names[match_index].upper()
            y1, x2, y2, x1 = location # (top, right, bottom, left)
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # 0.25로 축소 시킨 값을 다시 복원
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(image, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(image, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('WebCam', image)
    cv2.waitKey(1)
import os
import cv2
import numpy as np
import face_recognition

img = os.path.join(os.getcwd(), "data", "Elon Musk.jpg")
img_test = os.path.join(os.getcwd(), "data", "Elon Test.jpg")
# img_test = os.path.join(os.getcwd(), "data", "Bill Gates.jpg")

img_elon = face_recognition.load_image_file(img)
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file(img_test)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(img_elon)[0] # 1명 감지 (top, right, bottom, left)
encode_elon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon, (face_locations[3], face_locations[0]), (face_locations[1], face_locations[2]), (255,0,255), 2)

face_locations_test = face_recognition.face_locations(img_test)[0] # 1명 감지 (top, right, bottom, left)
encode_elon_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_locations_test[3], face_locations_test[0]), (face_locations_test[1], face_locations_test[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encode_elon], encode_elon_test)
face_distance = face_recognition.face_distance([encode_elon], encode_elon_test)
# print(results) # True면 같은사람
# print(face_distance)
cv2.putText(img_test, f'{results} {round(face_distance[0], 2)}%', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Test', img_test)
cv2.waitKey(0)
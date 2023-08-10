''' Face Detection from Image using OpenCV '''

import cv2

img = cv2.imread('Media/testImg.JPG')
# cv2.imshow('Image to Detect', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert BGR to Grayscale

FaceCascade = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')
faces = FaceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
# faces = FaceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f'Number of faces found = {len(faces)}')

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv2.imshow('Detected Faces', img)

# face_name = input("Enter face_name : ")
# face_id = input("Enter face_id : ")
# face_count = input("Enter count : ")

# cv2.imwrite("Dataset/" + str(face_name) + "/" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h, x:x+w])

cv2.waitKey(0)
import numpy as np
import cv2 as cv

faceCascade = cv.CascadeClassifier('./haar_face.xml')
cap = cv.VideoCapture(0)
cap.set(3, 640)         #set width
cap.set(4, 480)         #set height

while True:
    ret, frame = cap.read()
    img = cv.flip(frame, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2,
            minNeighbors=5, minSize = (20,20))
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]


    cv.imshow('Video', img)
    cv.imshow('Gray', gray)

    k = cv.waitKey(30) & 0xff
    if k == 27:                 # press 'ESC' to quit
        break


cap.release()
cv.destroyAllWindows()


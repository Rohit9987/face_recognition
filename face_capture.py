import cv2 as cv
import os

cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv.CascadeClassifier('./haar_face.xml')

# for each person, enter on numeric face id
face_id = input('\n Enter user id and press <return> ==> ')
print("\n [INFO] Initializing face capture. Look at the camera and wait...")

        
# initialize individual sampling dace count
count = 0
while True:
    ret, img = cam.read()
    img = cv.flip(img, 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        
        #save the captured image into the datasets folder
        cv.imwrite("./dataset/User." +str(face_id)+ "." +
                str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv.imshow('Image', img)
    
    k = cv.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >=99:
        break

# do a bit of clean up
print('\n {INFO] Exiting program and cleanup stuff')
cam.release()
cv.destroyAllWindows()
    


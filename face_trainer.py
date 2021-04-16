import cv2 as cv
import numpy as np
from PIL import Image
import os

# path for face image database
path = './dataset'
recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier('./haar_face.xml')

#function to get the images and label data
def getImagesandLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+h])
            ids.append(id)
    return faceSamples, ids

print('\n [INFO] Training faces. It will take a few seconds. Wait .. ')

faces, ids = getImagesandLabels(path)
recognizer.train(faces, np.array(ids))

# save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

#print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting program.".format(len(np.unique(ids))))

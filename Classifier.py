import sys
import os.path as op
import tensorflow as tf
import numpy as np
import cv2
from PIL import ImageGrab

TITLE = sys.argv[1]
labels, cols = np.loadtxt(op.join(TITLE,'Labels.tsv'), delimiter='\t', dtype='S20,S20', unpack=True)
IMG_SHAPE = (100, 128, 3)

model = tf.keras.models.load_model(op.join(TITLE,'model.h5'))
cascade_path = 'C:\\_data_\\lbpcascade_animeface.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def predict(img):
    img = cv2.resize(img, (np.max(IMG_SHAPE),np.max(IMG_SHAPE)))
    img = img[:IMG_SHAPE[0],:IMG_SHAPE[1],:]
    img = cv2.GaussianBlur(img, (3,3), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    predictions = model.predict(np.expand_dims(img,0))
    return np.argmax(predictions[0]), np.max(predictions[0])


#cv2.namedWindow('ret', cv2.WINDOW_NORMAL)
while cv2.waitKey(1)!=27:
    img = ImageGrab.grab(bbox=(100, 300, 100+1600, 300+900))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, minSize=(100,100), scaleFactor=1.05, minNeighbors=5)

    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        idx, val = predict(face)
        label = labels[idx].decode()
        col = tuple(map(int, cols[idx][1:-1].decode().split(',')))
        cv2.rectangle(img, (x,y), (x+w,y+h), col, 2)
        cv2.putText(img, '%s:%d%%'%(label,val*100), (x+10,y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

    cv2.imshow('Results', img)

cv2.destroyAllWindows()
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array

from tensorflow.keras.models import model_from_json
model = model_from_json(open(r"C:/Users/hp/Downloads/Emotion_Detection_CNN/model.json", "r").read())
model.load_weights(r'C:/Users/hp/Downloads/Emotion_Detection_CNN/model.h5')


import cv2
face_haar_cascade = cv2.CascadeClassifier(r'C:/Users/hp/Downloads/Emotion_Detection_CNN/haarcascade_frontalface_default.xml')



cap=cv2.VideoCapture(0)
while True:
    res,frame=cap.read()
    height, width , channel = frame.shape
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image )
    for (x,y, w, h) in faces:
        cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
        roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
        roi_gray=cv2.resize(roi_gray,(48,48))
        if np.sum([roi_gray]) != 0:
            #roi = roi_gray.astype('float') / 255.0
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            label_position = (x, y-10)
            cv2.putText(frame, emotion_prediction, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
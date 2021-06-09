import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

with open("network_emotions.json") as json_file:
    new_model = json_file.read()
# print(new_model)

loaded_model = keras.models.model_from_json(new_model)
loaded_model.load_weights("weights.hdf5")
loaded_model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
# print(loaded_model.summary())

camera= cv2.VideoCapture(0)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
while True:
    connected, frame = camera.read()
    if not connected:
        break
    else:
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.2, minSize=(100, 100))
        for detection in faces:
            x, y, w, h = detection
            roi = frame[y: y+h, x: x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = np.expand_dims(roi, axis=0)
            roi = roi/255
            probs = loaded_model.predict(roi)
            pred = np.argmax(probs)
            print(h, w)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotions[pred], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("FACE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
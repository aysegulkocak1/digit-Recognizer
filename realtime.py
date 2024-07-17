import os
import cv2
import numpy as np
from keras.models import load_model

# Disable GPU usage for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Recognizer:
    def __init__(self):
        self.model = load_model('digitmodels.h5')
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        reshaped = resized.reshape(1, 8, 8, 1)
        #reshaped = reshaped.astype('float32') / 255  # Normalize
        return reshaped

    def predict_digit(self, image):
        preprocessed = self.preprocess_image(image)
        prediction = self.model.predict(preprocessed)
        predicted_digit = np.argmax(prediction)
        return predicted_digit

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            prediction = self.predict_digit(frame)
            print(f"Predicted Digit: {prediction}")
            
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

recognizer = Recognizer()
recognizer.run()

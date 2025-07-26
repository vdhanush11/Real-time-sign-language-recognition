import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import math

# Load trained model and labels
model = tf.keras.models.load_model("hand_gesture_model.h5")
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Initialize webcam, hand detector, and TTS engine
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
tts = pyttsx3.init()

# Constants
imgSize = 300
offset = 20
predicted_word = ""
last_pred = ""
last_time = 0
delay = 2  # seconds

# Speak function
def speak(text):
    tts.say(text)
    tts.runAndWait()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    predicted_label = "?"

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        try:
            # Crop and resize
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Preprocess image for prediction
            imgInput = imgWhite / 255.0
            imgInput = np.expand_dims(imgInput, axis=0)  # Shape: (1, 300, 300, 3)

            prediction = model.predict(imgInput)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]
            predicted_label = labels[predicted_index]

            # Avoid repeating predictions too fast
            current_time = time.time()
            if confidence > 0.8 and (current_time - last_time > delay or predicted_label != last_pred):
                predicted_word += predicted_label
                last_pred = predicted_label
                last_time = current_time
                speak(predicted_label)

        except Exception as e:
            print("Error:", e)

    # Display results
    cv2.putText(img, f"Letter: {predicted_label}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(img, f"Word: {predicted_word}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("Sign Language Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        speak(f"The final word is {predicted_word}")
        break

cap.release()
cv2.destroyAllWindows()

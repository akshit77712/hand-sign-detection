import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time
import warnings
from cvzone.ClassificationModule import Classifier

# Suppress TensorFlow and protobuf warnings
warnings.filterwarnings('ignore')

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20  # Offset for cropping
imgSize = 300  # Size for the white square image

labels = ["a", "b", "c","d","e"]

folder = "data/c"  # Folder to save images
counter = 0

while True:
    success, img = cap.read()  # Capture frame from the webcam
    if not success:
        print("Failed to grab frame")
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]  # Get the first hand detected
        x, y, w, h = hand["bbox"]  # Bounding box of the hand

        # Create a white image of size 300x300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the cropping coordinates are within image bounds
        imgHeight, imgWidth, _ = img.shape

        # Correct boundaries for x and y
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)

        # Crop the image
        imgCrop = img[y1:y2, x1:x2]

        # Calculate the aspect ratio of the cropped image
        aspectRatio = h / w

        if aspectRatio > 1:
            # Height is greater than width, so resize based on height
            k = imgSize / h
            newWidth = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (newWidth, imgSize))  # Resize keeping height constant
            xOffset = (imgSize - newWidth) // 2  # Center horizontally
            imgWhite[:, xOffset:xOffset + newWidth] = imgResize

            # Pass the white background image to the classifier
            prediction, index = classifier.getPrediction(imgWhite)
            print(f"Prediction: {labels[index]}")
        else:
            # Width is greater than height, so resize based on width
            k = imgSize / w
            newHeight = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, newHeight))  # Resize keeping width constant
            yOffset = (imgSize - newHeight) // 2  # Center vertically
            imgWhite[yOffset:yOffset + newHeight, :] = imgResize

            # Pass the white background image to the classifier
            prediction, index = classifier.getPrediction(imgWhite)
            print(f"Prediction: {labels[index]}")

        # Display the cropped hand image and the white background image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame with hand landmarks
    cv2.imshow("Image", img)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

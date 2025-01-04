import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize webcam, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20  # Offset for cropping
imgSize = 300  # Size for the white square image
labels = ["a", "b", "c", "d", "e"]

# Tkinter GUI
class HandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Detection and Classification")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")  # Background color

        # Add title label
        self.title_label = Label(
            root,
            text="Hand Detection & Classification",
            font=("Helvetica", 24, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1",
        )
        self.title_label.pack(pady=20)

        # Create a frame to display the video
        self.video_frame = tk.Frame(root, bg="#34495e", width=800, height=500)
        self.video_frame.pack(pady=20)
        self.video_frame.pack_propagate(0)

        # Add a border to the video feed
        self.video_label = Label(self.video_frame, bg="#34495e")
        self.video_label.pack(expand=True)

        # Create a label to display the prediction
        self.prediction_label = Label(
            root,
            text="Prediction: ",
            font=("Helvetica", 20, "bold"),
            bg="#2c3e50",
            fg="#e74c3c",
        )
        self.prediction_label.pack(pady=10)

        # Create a button to close the application
        self.close_button = Button(
            root,
            text="Quit",
            font=("Helvetica", 16, "bold"),
            bg="#e74c3c",
            fg="#ecf0f1",
            activebackground="#c0392b",
            activeforeground="#ecf0f1",
            relief="groove",
            command=self.quit_app,
        )
        self.close_button.pack(pady=20)

        # Start the video loop
        self.update_frame()

    def update_frame(self):
        success, img = cap.read()
        if success:
            hands, img = detector.findHands(img)  # Detect hands

            if hands:
                hand = hands[0]
                x, y, w, h = hand["bbox"]

                # Create a white image of size 300x300
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                imgHeight, imgWidth, _ = img.shape
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)
                x2 = min(imgWidth, x + w + offset)
                y2 = min(imgHeight, y + h + offset)

                imgCrop = img[y1:y2, x1:x2]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    newWidth = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
                    xOffset = (imgSize - newWidth) // 2
                    imgWhite[:, xOffset:xOffset + newWidth] = imgResize
                else:
                    k = imgSize / w
                    newHeight = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                    yOffset = (imgSize - newHeight) // 2
                    imgWhite[yOffset:yOffset + newHeight, :] = imgResize

                # Get prediction
                prediction, index = classifier.getPrediction(imgWhite)
                self.prediction_label.config(text=f"Prediction: {labels[index]}")

            # Convert image for Tkinter
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgTK = ImageTk.PhotoImage(Image.fromarray(imgRGB))

            # Update video label
            self.video_label.imgtk = imgTK
            self.video_label.configure(image=imgTK)

        # Repeat after 10ms
        self.root.after(10, self.update_frame)

    def quit_app(self):
        cap.release()
        self.root.destroy()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root)
    root.mainloop()

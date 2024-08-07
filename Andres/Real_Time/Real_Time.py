import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow, QPushButton, QStatusBar
from ultralytics import YOLO
import tensorflow as tf

# Get the absolute path to the model files
yolo_model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-face.pt')
emotion_model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

# Load the YOLOv8 face detection model
facemodel = YOLO(yolo_model_path)

# Load the emotion detection model
emotion_model = tf.keras.models.load_model(emotion_model_path)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Create a label to display the video feed
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Arial', 10))

        # Create buttons for starting and stopping the camera
        self.start_button = QPushButton('Start Camera')
        self.start_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.start_button.setStyleSheet("background-color: #800080; color: white; padding: 10px; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton('Stop Camera')
        self.stop_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #800080; color: white; padding: 10px; border-radius: 5px;")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)  # Disable the stop button initially

        # Set up the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)

        # Initialize video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0  # Initialize frame counter

    def start_camera(self):
        # Start the camera
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  # Update every 20ms
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        # Stop the camera
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def preprocess_face(self, face):
        # Preprocess the face image for emotion detection model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face = cv2.resize(face, (48, 48))  # Resize to 48x48 pixels
        face = face.astype('float32') / 255  # Normalize pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        return face

    def update_frame(self):
        # Read frame from the camera
        ret, frame = self.cap.read()
        if ret:
            # Perform face detection using YOLOv8
            results = facemodel(frame)

            for result in results:
                boxes = result.boxes  # List of detected boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    label = facemodel.names[cls]
                    if label == "face":  # Check if the detected object is a face
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # Extract the face region
                        face = frame[int(y1):int(y2), int(x1):int(x2)]
                        if face.size > 0:
                            # Only perform emotion detection every 10 frames
                            if self.frame_count % 10 == 0:
                                face = self.preprocess_face(face)
                                # Predict the emotion
                                emotion_prediction = emotion_model.predict(face)
                                self.emotion_label = emotion_labels[np.argmax(emotion_prediction)]

                            # Display emotion label
                            cv2.putText(frame, self.emotion_label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Convert frame to QImage and display it
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

            # Increment frame counter
            self.frame_count += 1

    def closeEvent(self, event):
        # Ensure the camera is stopped when the window is closed
        self.stop_camera()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Face and Emotion Detection with YOLOv8")

        # Set the initial size of the main window
        self.resize(800, 600)

        # Set up the main video widget
        self.video_widget = VideoWidget()
        self.setCentralWidget(self.video_widget)

        # Add a status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Apply stylesheets for dark mode and styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
                color: #ffffff;
            }
            QLabel {
                font-size: 14px;
                border: 1px solid #ccc;
                background-color: #424242;
                color: #ffffff;
                padding: 10px;
            }
            QPushButton {
                font-size: 14px;
                padding: 10px;
                background-color: #800080;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QPushButton:hover {
                background-color: #9932CC;
            }
            QStatusBar {
                background-color: #333333;
                color: #ffffff;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

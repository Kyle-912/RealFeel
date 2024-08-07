import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMainWindow, QPushButton, QMenuBar, QAction, QStatusBar
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
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Arial', 10))

        self.start_button = QPushButton('Start Camera')
        self.start_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.start_button.setStyleSheet("background-color: #800080; color: white; padding: 10px; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton('Stop Camera')
        self.stop_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #800080; color: white; padding: 10px; border-radius: 5px;")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        return face

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = facemodel(frame)  # Perform detection

            for result in results:
                boxes = result.boxes  # Assuming boxes is a list of Box objects
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the bounding box
                    conf = box.conf[0]  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    label = facemodel.names[cls]
                    if label == "face":  # Adjust based on your model's label for face
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        face = frame[int(y1):int(y2), int(x1):int(x2)]
                        if face.size > 0:
                            face = self.preprocess_face(face)
                            emotion_prediction = emotion_model.predict(face)
                            emotion_label = emotion_labels[np.argmax(emotion_prediction)]
                            cv2.putText(frame, emotion_label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def closeEvent(self, event):
        self.stop_camera()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Face and Emotion Detection with YOLOv8")

        self.video_widget = VideoWidget()
        self.setCentralWidget(self.video_widget)

        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('File')
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

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
            QMenuBar {
                background-color: #333333;
                color: white;
            }
            QMenuBar::item {
                background-color: #333333;
                color: white;
            }
            QMenuBar::item::selected {
                background-color: #555555;
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

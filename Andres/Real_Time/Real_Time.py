import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
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

class VideoThread(QThread):
    update_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(VideoThread, self).__init__(parent)
        self.cap = None
        self.running = False

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.update_frame_signal.emit(frame)
            QThread.msleep(50)  # Reduce frame rate to approximately 20 FPS

    def stop(self):
        self.running = False
        self.wait()
        if self.cap:
            self.cap.release()

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Arial', 10))

        self.start_button = QPushButton('Start Camera')
        self.start_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton('Stop Camera')
        self.stop_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; border-radius: 5px;")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)  # Disable the stop button initially

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.setLayout(self.layout)

        self.video_thread = VideoThread()
        self.video_thread.update_frame_signal.connect(self.update_frame)

        self.frame_count = 0  # Initialize frame counter

    def start_camera(self):
        self.video_thread.running = True
        self.video_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_camera(self):
        self.video_thread.stop()
        self.image_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def preprocess_face(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face = cv2.resize(face, (48, 48))  # Resize to 48x48 pixels
        face = face.astype('float32') / 255  # Normalize pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        return face

    def update_frame(self, frame):
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
                    color = (255, 255, 0)  # Cyan color for the face bounding box
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

                        # Display emotion label with outline
                        text_size, _ = cv2.getTextSize(self.emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_w, text_h = text_size
                        label_y = int(y1) - text_h - 20  # Raise the label position a bit higher
                        if label_y < 0:  # Ensure the label is within the frame
                            label_y = 0
                        cv2.rectangle(frame, (int(x1), label_y), (int(x1) + text_w, label_y + text_h + 10), (0, 0, 0), -1)  # Background for better visibility
                        cv2.putText(frame, self.emotion_label, (int(x1), label_y + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # Text with outline

        # Convert frame to QImage and display it
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

        # Increment frame counter
        self.frame_count += 1

    def closeEvent(self, event):
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
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton#stop_button {
                background-color: #f44336;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#stop_button:hover {
                background-color: #d32f2f;
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
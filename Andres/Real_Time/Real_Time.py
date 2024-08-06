import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
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
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def preprocess_face(self, face):
        # Assuming your model expects 48x48 grayscale images
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        return face

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # YOLO object detection
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
                        
                        # Extract the face region
                        face = frame[int(y1):int(y2), int(x1):int(x2)]
                        if face.size > 0:
                            face = self.preprocess_face(face)
                            
                            # Predict the emotion
                            emotion_prediction = emotion_model.predict(face)
                            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

                            # Display emotion label
                            cv2.putText(frame, emotion_label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Convert frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, aspectRatioMode=True)
            self.image_label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoWidget()
    win.setWindowTitle("Real-time Face and Emotion Detection with YOLOv8")
    win.show()
    sys.exit(app.exec_())

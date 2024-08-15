import torch
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import threading
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

cap = cv2.VideoCapture(0)

class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()
        self.square_size = 300  # Kare boyutu
        threading.Thread(target=self.yolo).start()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1152, 837)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(280, 20, 640, 640))
        self.camera.setObjectName("camera")
        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(390, 730, 141, 16))
        self.text.setObjectName("text")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(530, 760, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(530, 730, 113, 25))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1152, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.camera.setText(_translate("MainWindow", ""))
        self.text.setText(_translate("MainWindow", "KARENİN BOYUTU :"))
        self.pushButton.setText(_translate("MainWindow", "DEĞİŞTİR"))
        self.pushButton.clicked.connect(self.change_square_size)

    def change_square_size(self):
        try:
            new_size = int(self.lineEdit.text())
            self.square_size = new_size
        except ValueError:
            pass

    def yolo(self):
        while cap.isOpened():
            self.ret, self.frame = cap.read()
            if not self.ret:
                break

            # Görüntüyü 640x640 boyutuna yeniden boyutlandırın
            self.frame_resized = cv2.resize(self.frame, (640, 640))

            self.results = model(self.frame_resized)
            self.frame = self.results.render()[0]

            self.kareciz()
            self.cizgi_cikar()
            self.check_inside()

            # OpenCV'den PyQt'ye görüntü aktarımı
            self.display_image(self.frame)

        # Release video capture and close windows
        cap.release()

    def kareciz(self):
        # Create a writable copy of the frame
        self.frame = self.frame.copy()

        # Get frame dimensions
        height, width, _ = self.frame.shape

        # Calculate the top-left and bottom-right coordinates of the square
        self.top_left_x = (width - self.square_size) // 2
        self.top_left_y = (height - self.square_size) // 2
        self.bottom_right_x = self.top_left_x + self.square_size
        self.bottom_right_y = self.top_left_y + self.square_size

        # Draw the square in the center of the frame
        cv2.rectangle(self.frame, (self.top_left_x, self.top_left_y), (self.bottom_right_x, self.bottom_right_y), (255, 0, 0), 2)

    def check_inside(self):
        for det in self.results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det

            # Check if the center of the bounding box is inside the square
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if (self.top_left_x <= center_x <= self.bottom_right_x) and (
                    self.top_left_y <= center_y <= self.bottom_right_y):
                # Draw "INSIDE" text on the frame
                cv2.putText(self.frame, "INSIDE", (self.top_left_x + 10, self.top_left_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(self.frame, "OUTSIDE", (self.top_left_x + 10, self.top_left_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 100, 100), 2)

    def cizgi_cikar(self):
        # Get frame dimensions
        height, width, _ = self.frame.shape

        # Determine the center of the frame
        center_x = width // 2
        center_y = height // 2

        # Draw lines from the center to each detected object's center
        for det in self.results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det

            # Calculate the center of the detected object
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # Draw a line from the center of the frame to the center of the detected object
            cv2.line(self.frame, (center_x, center_y), (int(object_center_x), int(object_center_y)), (0, 255, 0), 2)

    def display_image(self, img):
        # Convert the frame to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to QImage
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # Set the image to the QLabel
        self.camera.setPixmap(QtGui.QPixmap.fromImage(q_img))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

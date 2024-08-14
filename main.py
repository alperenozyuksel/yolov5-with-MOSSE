import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

# Open video file or capture from webcam
cap = cv2.VideoCapture('640640.mp4')  # Replace with 0 to use webcam

class Dogfight:
    def __init__(self):
        self.yolo()

    def yolo(self):
        while cap.isOpened():
            self.ret, self.frame = cap.read()
            if not self.ret:
                break

            self.results = model(self.frame)
            self.frame = self.results.render()[0]

            self.kareciz()

            # Draw lines from the center to detected objects
            self.cizgi_cikar()
            self.check_inside()

            cv2.imshow('YOLOv5 Video', self.frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and close windows
        cap.release()
        cv2.destroyAllWindows()

    def kareciz(self):
        # Create a writable copy of the frame
        self.frame = self.frame.copy()

        # Define the size of the square (for example, 300x300 pixels)
        square_size = 300

        # Get frame dimensions
        height, width, _ = self.frame.shape

        # Calculate the top-left and bottom-right coordinates of the square
        self.top_left_x = (width - square_size) // 2
        self.top_left_y = (height - square_size) // 2
        self.bottom_right_x = self.top_left_x + square_size
        self.bottom_right_y = self.top_left_y + square_size

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
                # Draw "İçerde" text on the frame
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

Dogfight()

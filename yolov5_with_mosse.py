import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

# Open video file or capture from webcam
cap = cv2.VideoCapture('640640.mp4')  # Replace with 0 to use webcam

class Dogfight:
    def __init__(self):
        self.mosse_tracker = None  # Initialize MOSSE tracker as None
        self.yolo()

    def yolo(self):
        while cap.isOpened():
            self.ret, self.frame = cap.read()
            if not self.ret:
                break

            if self.mosse_tracker is None:
                # Run YOLO detection
                self.results = model(self.frame)
                detections = self.results.xyxy[0]

                if len(detections) > 0:
                    # Assuming we're tracking the first detected object
                    x1, y1, x2, y2, conf, cls = detections[0].numpy()
                    bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))

                    # Initialize MOSSE tracker with the first detected object's bounding box
                    self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
                    self.mosse_tracker.init(self.frame, bbox)

            else:
                # Update MOSSE tracker
                success, bbox = self.mosse_tracker.update(self.frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # If tracking fails, reset the tracker and switch back to YOLO detection
                    self.mosse_tracker = None

            self.kareciz()

            # Draw lines from the center to detected objects if any
            if self.mosse_tracker is None:
                self.cizgi_cikar()

            # Check if tracked object is inside the square
            if self.mosse_tracker is not None:
                self.check_inside(bbox)

            cv2.imshow('YOLOv5 Video', self.frame)

            # Exit on 'q' key press
            if cv2.waitKey(30) & 0xFF == ord('q'):
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

    def check_inside(self, bbox):
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2

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
            x1, y1, x2, y2, conf, cls = det.numpy()

            # Calculate the center of the detected object
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # Draw a line from the center of the frame to the center of the detected object
            cv2.line(self.frame, (center_x, center_y), (int(object_center_x), int(object_center_y)), (0, 255, 0), 2)

Dogfight()

import torch
import cv2
import time
import threading
from cv2 import resize

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/deneme.onnx')

cap = cv2.VideoCapture('videos/640640.mp4')

class Dogfight:
    def __init__(self):
        self.mosse_tracker = None
        self.tracking_active = False
        self.yolo_active = True
        self.frame = None
        self.ret = None
        self.results = None

        # Start threading
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.processing_thread = threading.Thread(target=self.yolo)

        self.capture_thread.start()
        self.processing_thread.start()

        self.capture_thread.join()
        self.processing_thread.join()

    def capture_frames(self):
        while cap.isOpened():
            self.ret, self.frame = cap.read()
            if not self.ret:
                break

            self.frame = resize(self.frame, (640, 640))
            self.kareciz()

            time.sleep(0.03)  # Mimic cv2.waitKey(30)

        cap.release()

    def yolo(self):
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_display = 0

        while True:
            if self.frame is not None:
                if self.tracking_active and self.mosse_tracker is not None:
                    self.mosse()
                else:
                    if self.yolo_active:
                        self.run_yolo()

                self.fps()

                cv2.imshow('YOLOv5 with MOSSE', self.frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_tracking()

        cv2.destroyAllWindows()

    def kareciz(self):
        self.frame = self.frame.copy()

        square_size = 300
        height, width, _ = self.frame.shape

        self.top_left_x = (width - square_size) // 2
        self.top_left_y = (height - square_size) // 2
        self.bottom_right_x = self.top_left_x + square_size
        self.bottom_right_y = self.top_left_y + square_size

        cv2.rectangle(self.frame, (self.top_left_x, self.top_left_y), (self.bottom_right_x, self.bottom_right_y), (255, 0, 0), 2)

    def check_inside(self, bbox):
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2

        if (self.top_left_x <= center_x <= self.bottom_right_x) and (
                self.top_left_y <= center_y <= self.bottom_right_y):
            cv2.putText(self.frame, "INSIDE", (self.top_left_x + 10, self.top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(self.frame, "OUTSIDE", (self.top_left_x + 10, self.top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 100, 100), 2)

    def cizgi_cikar(self):
        height, width, _ = self.frame.shape

        center_x = width // 2
        center_y = height // 2

        for det in self.results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.numpy()

            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            cv2.line(self.frame, (center_x, center_y), (int(object_center_x), int(object_center_y)), (0, 255, 0), 2)

    def mosse(self):
        if self.mosse_tracker is not None:
            success, bbox = self.mosse_tracker.update(self.frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.check_inside(bbox)
            else:
                self.reset_tracking()

    def run_yolo(self):
        self.results = model(self.frame)
        detections = self.results.xyxy[0]

        if len(detections) > 0:
            x1, y1, x2, y2, conf, cls = detections[0].numpy()

            bbox_padding = 10
            x1 = max(0, x1 - bbox_padding)
            y1 = max(0, y1 - bbox_padding)
            x2 = min(self.frame.shape[1], x2 + bbox_padding)
            y2 = min(self.frame.shape[0], y2 + bbox_padding)

            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
            self.mosse_tracker.init(self.frame, bbox)
            self.tracking_active = True
            self.yolo_active = False

        self.cizgi_cikar()

    def fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        if elapsed_time >= 1.0:
            self.fps_display = self.frame_count / elapsed_time
            self.frame_count = 0
            self.prev_time = current_time

        cv2.putText(self.frame, f'FPS: {int(self.fps_display)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def reset_tracking(self):
        self.mosse_tracker = None
        self.tracking_active = False
        self.yolo_active = True

Dogfight()

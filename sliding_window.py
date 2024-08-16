import torch
import cv2
import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

# Open video file or capture from webcam
cap = cv2.VideoCapture('640640.mp4')  # Replace with 0 to use webcam

class Dogfight:
    def __init__(self):
        self.region_index = 0  # Başlangıç bölgesi
        self.regions = self.define_regions()
        self.start_time = time.time()
        self.yolo()

    def define_regions(self):
        regions = []
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        region_width, region_height = width // 3, height // 2

        for i in range(2):  # 2 satır
            for j in range(3):  # 3 sütun
                x_start = j * region_width
                y_start = i * region_height
                x_end = x_start + region_width
                y_end = y_start + region_height
                regions.append((x_start, y_start, x_end, y_end))

        return regions

    def yolo(self):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mevcut bölgeyi seç
            x_start, y_start, x_end, y_end = self.regions[self.region_index]
            region = frame[y_start:y_end, x_start:x_end]

            # Bölgeyi modelin beklediği boyutlara yeniden boyutlandır
            resized_region = cv2.resize(region, (640, 640))

            # YOLOv5 modelini seçilen bölgede uygula
            results = model(resized_region)
            output_region = results.render()[0]

            # İşlenmiş bölgeyi orijinal boyutlara geri döndür
            output_region = cv2.resize(output_region, (x_end - x_start, y_end - y_start))

            # İşlenmiş bölgeyi orijinal kareye geri ekle
            frame[y_start:y_end, x_start:x_end] = output_region

            # Tarama bölgesini kutu içinde göster
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            # Kareyi göster
            cv2.imshow('YOLOv5 Video', frame)

            # Bölgeyi saniyede bir değiştir
            if time.time() - self.start_time >= 0.10:
                self.region_index = (self.region_index + 1) % len(self.regions)
                self.start_time = time.time()

            # 'q' tuşuna basıldığında çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Video yakalamayı serbest bırak ve pencereleri kapat
        cap.release()
        cv2.destroyAllWindows()

Dogfight()

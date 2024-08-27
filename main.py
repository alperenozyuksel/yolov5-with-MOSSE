import torch
import cv2

# Load the YOLOv5 model from the provided ONNX file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/deneme.onnx')

# Open the video file
cap = cv2.VideoCapture('videos/640640.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('YOLOv5 Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

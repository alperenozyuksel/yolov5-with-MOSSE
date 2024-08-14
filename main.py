import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

# Open video file or capture from webcam
cap = cv2.VideoCapture('640640.mp4')  # Replace with 0 to use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]

    # Display the result frame-by-frame
    cv2.imshow('YOLOv5 Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

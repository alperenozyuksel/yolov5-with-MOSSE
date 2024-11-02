import torch
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# YOLOv5 modelini yükleme
model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx')

# ROS düğümünü başlatma
rospy.init_node('yolo_camera_integration', anonymous=True)

# CvBridge nesnesi, ROS görüntüsünü OpenCV formatına çevirmek için kullanılır
bridge = CvBridge()

def image_callback(msg):
    # ROS Image mesajını OpenCV görüntüsüne dönüştürme
    frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

    # Görüntüyü modele uygun olacak şekilde 640x640 boyutuna yeniden boyutlandırma
    frame_resized = cv2.resize(frame, (640, 640))

    # YOLO modelinde tahmin çalıştırma
    results = model(frame_resized)

    # Tahmin sonuçlarını görüntüye yerleştirme
    frame_resized = results.render()[0]

    # Sonuçları gösterme
    cv2.imshow("YOLOv5 ROS Camera", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("Kapatıldı.")

# /camera/rgb/image_raw konusundan görüntüleri alma
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)

# ROS döngüsünü başlatma
rospy.spin()

# Ekran penceresini kapatma
cv2.destroyAllWindows()


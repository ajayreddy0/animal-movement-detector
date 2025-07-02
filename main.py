# main.py
from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

# Load YOLOv8 model (auto downloads if not present)
model = YOLO("yolov8n.pt")

# Animal classes (COCO-trained)
animal_classes = ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Define restricted zone (adjust to your camera view)
zone_top_left = (100, 100)
zone_bottom_right = (500, 400)

alert_triggered = False

# Function to play alert sound in separate thread
def play_alert():
    playsound('alert.mp3')

def is_in_zone(x, y):
    return zone_top_left[0] < x < zone_bottom_right[0] and zone_top_left[1] < y < zone_bottom_right[1]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        if class_name in animal_classes and confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if is_in_zone(center_x, center_y):
                cv2.putText(frame, "INTRUSION!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if not alert_triggered:
                    alert_triggered = True
                    threading.Thread(target=play_alert, daemon=True).start()
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw restricted zone
    cv2.rectangle(frame, zone_top_left, zone_bottom_right, (255, 0, 0), 2)
    cv2.putText(frame, "Restricted Zone", (zone_top_left[0], zone_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Animal Movement Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        alert_triggered = False  # Reset alert manually

cap.release()
cv2.destroyAllWindows()

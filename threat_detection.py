import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Download the model weights from Hugging Face
model_path = hf_hub_download(
    repo_id="Subh775/Threat-Detection-YOLOv8n",
    filename="weights/best.pt"
)

# 2. Load the YOLOv8 threat detection model
model = YOLO(model_path)

# Label names expected from this model
# Classes: 0=Gun, 1=Explosive, 2=Grenade, 3=Knife
labels = model.names

# 3. Open webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.4.1:81/stream")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.3)

    # Draw boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = labels.get(cls_id, f"class{cls_id}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label
            text = f"{label.upper()} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Threat Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

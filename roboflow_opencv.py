import numpy as np
import cv2 as cv
from roboflow import Roboflow

rf = Roboflow(api_key="VKi8pA7O3w2d15dgVoDs")
project = rf.workspace().project("yolo-waste-detection")
model = project.version(1).model

cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# frame = cv.imread("image.jpg")
# ret = 1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Perform object detection
    predictions = model.predict(frame, confidence=40, overlap=30)
    predictions = predictions.json()

    # Draw bounding boxes on the frame
    for prediction in predictions["predictions"]:
        x, y, width, height = (
            int(prediction["x"]),
            int(prediction["y"]),
            int(prediction["width"]),
            int(prediction["height"]),
        )
        x, y = int(x - width/2), int(y - height/2)
        confidence = prediction["confidence"]
        class_name = prediction["class"]

        # Draw bounding box
        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display class and confidence
        text = f"{class_name}: {confidence:.2f}"
        cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
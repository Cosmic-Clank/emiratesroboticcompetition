from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import math

class ObjectDetection:
    def __init__(self, path_to_model, labels):
        self.model = YOLO(path_to_model)
        self.labels = labels
        
    def get_infered_image(self, np_image, depth_frame):
        results = self.model.predict(np_image, verbose=False)
        infered_image = np_image.copy()
        
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(infered_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", self.labels[cls])

                # object details
                org = [x1, y1 - 5]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.6
                color = (255, 0, 0)
                thickness = 2
                center = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))
                
                distance = depth_frame.get_distance(center[0], center[1])
                cv2.circle(infered_image, center, 5, (0, 0, 255), -1)

                cv2.rectangle(infered_image, (x1, y1 - 20), (x2, y1), (255, 0, 255), -1)
                cv2.putText(infered_image, f"{self.labels[cls].upper()} CONF: {confidence} CORDS: {center} DIST: {distance:.2f}m", org, font, fontScale, color, thickness)

        return infered_image
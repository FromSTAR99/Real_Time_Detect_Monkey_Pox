import cv2
from ultralytics import YOLO
import serial

# Load the YOLOv8 model
model = YOLO('C:\\Users\\gameh\\OneDrive\\Masaüstü\\Biyomedikal\\best_2.pt')

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,conf=0.5)
        

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Inference", annotated_frame)

   
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
       
        break


cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Initialize results dictionary and SORT tracker
results = {}
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('yolov8n.pt')  # COCO model for detecting vehicles
license_plate_detector = YOLO('last.pt')  # Model for detecting license plates

# Load video
cap = cv2.VideoCapture('sample1.mp4')

# Define video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_detections.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define vehicle classes (2: Car, 3: Motorcycle, 5: Bus, 7: Truck)
vehicles = [2, 3, 5, 7]

# Read video frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect vehicles using the COCO model
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                # Draw vehicle bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle ID: {class_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track vehicles using SORT
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop the license plate from the frame
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate: convert to grayscale and thresholding
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

                    # Draw bounding box for license plate and overlay OCR text
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'Plate: {license_plate_text}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

        # Save the frame with drawn detections and OCR results to the video
        out.write(frame)

# Write results to CSV
write_csv(results, 'test.csv')

# Release video capture and writer, and close windows
cap.release()
out.release()

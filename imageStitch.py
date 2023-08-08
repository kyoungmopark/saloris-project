import cv2
import numpy as np
import torch
from sort.sort import Sort

cap1 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
cap2 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

tracker = Sort()
stitcher = cv2.Stitcher_create()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    images = [frame1, frame2]

    status, stitched = stitcher.stitch(images)
    if stitched is None:
        print("Stitching failed.")
        continue
    pred = model(stitched)
    boxes = pred.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids = pred.xyxy[0].cpu().numpy()[:, 5].astype(int) 

    detections = np.column_stack((boxes, class_ids))
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x, y, w, h, track_id = [int(v) for v in obj]
        cv2.rectangle(stitched, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(stitched, str(track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    stitched = cv2.resize(stitched, None, fx=0.3, fy=0.3)

    cv2.imshow('Stitched Frame', stitched)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
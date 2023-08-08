import cv2
import numpy as np
import torch
from sort.sort import Sort

cap1 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
cap2 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

tracker = Sort()

orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

M = None  
Ht = None  

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    frame1 = cv2.resize(frame1, None, fx=0.4, fy=0.4)
    frame2 = cv2.resize(frame2, None, fx=0.4, fy=0.4)
    
    if M is None or Ht is None:
        kp1, des1 = orb.detectAndCompute(frame1, None)
        kp2, des2 = orb.detectAndCompute(frame2, None)
        
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7)

        corners1 = np.float32([[0, 0], [0, frame1.shape[0]], [frame1.shape[1], frame1.shape[0]], [frame1.shape[1], 0]]).reshape(-1, 1, 2)
        corners2 = cv2.perspectiveTransform(np.float32([[0, 0], [0, frame2.shape[0]], [frame2.shape[1], frame2.shape[0]], [frame2.shape[1], 0]]).reshape(-1, 1, 2), M)
        corners = np.vstack((corners1, corners2))
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
        t = [-x_min, -y_min]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  

    stitched = cv2.warpPerspective(frame1, Ht.dot(M), (x_max - x_min, y_max - y_min))
    stitched[t[1]:frame2.shape[0]+t[1], t[0]:frame2.shape[1]+t[0]] = frame2

    pred = model(stitched)
    boxes = pred.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids = pred.xyxy[0].cpu().numpy()[:, 5].astype(int) 

    detections = np.column_stack((boxes, class_ids))
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x, y, x_plus_w, y_plus_h, track_id = [int(v) for v in obj]
        cv2.rectangle(stitched, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        cv2.putText(stitched, str(track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Stitched Frame', stitched)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
import cv2
import numpy as np
import torch
import threading
from DeepSORT_YOLOv5_Pytorch.deep_sort import DeepSort
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time

class VideoCaptureThreaded:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.thread = threading.Thread(target=self.update, args=())

    def start(self):
        self.thread.start()

    def update(self):
        while self.is_running:
            if self.cap.isOpened():
                (self.ret, self.frame) = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.is_running = False
        self.thread.join()
        
    def get(self, prop_id):
        return self.cap.get(prop_id)

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


def match_objects(galary, id_dict,global_id):
    data = []
    clustering = DBSCAN(eps=1000, min_samples=2)
    pca = PCA(n_components=1)
    for obj in galary:    
        pca_obj = pca.fit_transform(obj[1])
        if len(pca_obj) > 2:
            flattened_data = [item for sublist in pca_obj for item in sublist]
            data.append(flattened_data)
    if len(data) > 0:
        clustering.fit(data)
        for label in set(clustering.labels_):
            if label == -1: 
                continue
            for i, obj_label in enumerate(clustering.labels_):
                if obj_label == label:
                    galary[i] = galary[i] + (global_id,) 
                    if galary[i][0] not in id_dict:  
                        id_dict[galary[i][0]] = global_id
                        global_id += 1
    return galary, id_dict

def process_tracked_objects(galary, tracked_objects, cam_id):
    for d in tracked_objects:
        xmin, ymin, xmax, ymax, id, vx,vy= d
        center = ((xmin+xmax)/2, (ymin+ymax)/2)
        key = f"{cam_id}_{id}"
        if abs(vx) > 0.5 or abs(vy) > 0.5:
            moving = True
        else:
            moving = False
        galary.append(key)
    return galary



cap1 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp")
cap2 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")
cap3 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
cap4 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp")
cap5 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.111/profile2/media.smp")
cap6 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.115/profile2/media.smp")
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
tracker1 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker2 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker3 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker4 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker5 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker6 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
global_id = 0
id_dict = {}
cap1.start()
cap2.start()
cap3.start()
cap4.start()
cap5.start()
cap6.start()
num_frames = 0
start_time = time.time()
trajectory = {}
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    ret5, frame5 = cap5.read()
    ret6, frame6 = cap6.read()
    frame1 = cv2.resize(frame1, None, fx=0.3, fy=0.3)
    frame2 = cv2.resize(frame2, None, fx=0.3, fy=0.3)       
    frame3 = cv2.resize(frame3, None, fx=0.3, fy=0.3)
    frame4 = cv2.resize(frame4, None, fx=0.3, fy=0.3)       
    frame5 = cv2.resize(frame5, None, fx=0.3, fy=0.3) 
    frame6 = cv2.resize(frame6, None, fx=0.3, fy=0.3) 
    pred1 = model(frame1)
    pred2 = model(frame2)
    pred3 = model(frame3)
    pred4 = model(frame4)
    pred5 = model(frame5)
    pred6 = model(frame6)
    boxes1 = pred1.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids1 = pred1.xyxy[0].cpu().numpy()[:, 5].astype(int) 
    boxes2 = pred2.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids2 = pred2.xyxy[0].cpu().numpy()[:, 5].astype(int)
    boxes3 = pred3.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids3 = pred3.xyxy[0].cpu().numpy()[:, 5].astype(int) 
    boxes4 = pred4.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids4 = pred4.xyxy[0].cpu().numpy()[:, 5].astype(int)
    boxes5 = pred5.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids5 = pred5.xyxy[0].cpu().numpy()[:, 5].astype(int) 
    boxes6 = pred6.xyxy[0].cpu().numpy()[:, :4].astype(int) 
    class_ids6 = pred6.xyxy[0].cpu().numpy()[:, 5].astype(int)
    boxes1[:, 2] -= boxes1[:, 0]
    boxes1[:, 3] -= boxes1[:, 1]
    boxes1[:, 0] = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1[:, 1] = boxes1[:, 1] + boxes1[:, 3] / 2
    boxes2[:, 2] -= boxes2[:, 0]
    boxes2[:, 3] -= boxes2[:, 1]
    boxes2[:, 0] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2[:, 1] = boxes2[:, 1] + boxes2[:, 3] / 2
    boxes3[:, 2] -= boxes3[:, 0]
    boxes3[:, 3] -= boxes3[:, 1]
    boxes3[:, 0] = boxes3[:, 0] + boxes3[:, 2] / 2
    boxes3[:, 1] = boxes3[:, 1] + boxes3[:, 3] / 2
    boxes4[:, 2] -= boxes4[:, 0]
    boxes4[:, 3] -= boxes4[:, 1]
    boxes4[:, 0] = boxes4[:, 0] + boxes4[:, 2] / 2
    boxes4[:, 1] = boxes4[:, 1] + boxes4[:, 3] / 2
    boxes5[:, 2] -= boxes5[:, 0]
    boxes5[:, 3] -= boxes5[:, 1]
    boxes5[:, 0] = boxes5[:, 0] + boxes5[:, 2] / 2
    boxes5[:, 1] = boxes5[:, 1] + boxes5[:, 3] / 2
    boxes6[:, 2] -= boxes6[:, 0]
    boxes6[:, 3] -= boxes6[:, 1]
    boxes6[:, 0] = boxes6[:, 0] + boxes6[:, 2] / 2
    boxes6[:, 1] = boxes6[:, 1] + boxes6[:, 3] / 2
    tracked_objects1 = tracker1.update(boxes1, class_ids1, frame1)
    tracked_objects2 = tracker2.update(boxes2, class_ids2, frame2)
    tracked_objects3 = tracker3.update(boxes3, class_ids3, frame3)
    tracked_objects4 = tracker4.update(boxes4, class_ids4, frame4)
    tracked_objects5 = tracker5.update(boxes5, class_ids5, frame5)
    tracked_objects6 = tracker6.update(boxes6, class_ids6, frame6)
    galary = []
    galary = process_tracked_objects(galary, tracked_objects1, cam_id=1)
    galary = process_tracked_objects(galary,tracked_objects2, cam_id=2)
    galary = process_tracked_objects(galary,tracked_objects3, cam_id=3)
    galary = process_tracked_objects(galary,tracked_objects4, cam_id=4)
    galary = process_tracked_objects(galary,tracked_objects5, cam_id=5)
    galary = process_tracked_objects(galary,tracked_objects6, cam_id=6)
    #if galary:
        #galary,id_dict = match_objects(galary,id_dict,global_id)

    for d in tracked_objects1:
        xmin, ymin, xmax, ymax, id,vx,vy = d
        key = f"1_{id}" 
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame1, (xmin ,ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame1, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for d in tracked_objects2:
        xmin, ymin, xmax, ymax, id, vx,vy = d
        key = f"2_{id}"
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame2, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for d in tracked_objects3:
        xmin, ymin, xmax, ymax, id, vx,vy = d
        key = f"3_{id}"
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame3, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame3, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for d in tracked_objects4:
        xmin, ymin, xmax, ymax, id,vx,vy = d
        key = f"4_{id}" 
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame4, (xmin ,ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame4, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for d in tracked_objects5:
        xmin, ymin, xmax, ymax, id, vx,vy = d
        key = f"5_{id}"
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame5, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame5, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for d in tracked_objects6:
        xmin, ymin, xmax, ymax, id,vx,vy = d
        key = f"6_{id}" 
        if key in id_dict:
            id = id_dict[key]
            cv2.rectangle(frame6, (xmin ,ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame6, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Frame1', frame1)
    cv2.imshow('Frame2', frame2)
    cv2.imshow('Frame3', frame3)
    cv2.imshow('Frame4', frame4)
    cv2.imshow('Frame5', frame5)
    cv2.imshow('Frame6', frame6)
    num_frames += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
processing_time = end_time - start_time
fps = num_frames / processing_time
print(f"Processing speed: {fps} frames per second")
cap1.stop()
cap2.stop()
cap3.stop()
cap4.stop()
cap5.stop()
cap6.stop()
cv2.destroyAllWindows()
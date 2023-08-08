import cv2
import numpy as np
import torch
import threading
from DeepSORT_YOLOv5_Pytorch.deep_sort import DeepSort
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
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

def cosine_similarity(feature1, feature2):
    channel_similarities = []
    for i, (channel_feature1, channel_feature2) in enumerate(zip(feature1, feature2)):
        channel_feature1 = channel_feature1 
        channel_feature2 = channel_feature2 
        
        norm1 = np.linalg.norm(channel_feature1)
        norm2 = np.linalg.norm(channel_feature2)

        channel_feature1_norm = channel_feature1 / norm1
        channel_feature2_norm = channel_feature2 / norm2

        channel_similarity = np.dot(channel_feature1_norm, channel_feature2_norm)
        channel_similarities.append(channel_similarity)

    average_similarity = np.mean(channel_similarities)
    return average_similarity

def calculate_center(bbox):
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

def is_inside(center, overlap_section):
    dist = cv2.pointPolygonTest(overlap_section, center, False)
    return dist >= 0

def transform_bbox(bbox, warp_matrix, flip=False):
    bbox_h = np.array([bbox[0], bbox[1], 1])
    bbox_transformed_h = np.dot(warp_matrix, bbox_h)
    bbox_transformed = bbox_transformed_h[:2] / bbox_transformed_h[2]
    return bbox_transformed

def calculate_cost_matrix(overlap_objects1_transformed, overlap_objects2_transformed, lambda_weight=0.4):
    cost_matrix = np.zeros((len(overlap_objects1_transformed), len(overlap_objects2_transformed)))
    scale_factor = 2 / np.sqrt(500**2 + 500**2)
    for i, (bbox1, key1, feature1) in enumerate(overlap_objects1_transformed):
        for j, (bbox2, key2, feature2) in enumerate(overlap_objects2_transformed):
            spatial_distance = scale_factor * np.linalg.norm(np.array(bbox1) - np.array(bbox2))
            if not feature1 or not feature2 :
                feature_distance = 1
            else:
                feature1_np = np.array(feature1)
                feature2_np = np.array(feature2)
                feature_distance = 1- cosine_similarity(feature1_np, feature2_np)
            cost_matrix[i][j] = lambda_weight * spatial_distance + (1 - lambda_weight) * feature_distance 

    return cost_matrix


def match_objects(overlap_objects1, overlap_objects2, overlap_objects3,overlap_objects4, frame1_warp_mat, frame2_warp_mat,frame3_warp_mat,frame4_warp_mat,max_distance =1):
    
    global global_id
    if not overlap_objects1 or not overlap_objects2 or not overlap_objects3 or not overlap_objects4: 
        return []
    overlap_objects1_transformed = [(transform_bbox(center, frame1_warp_mat), key, feature) for center, key, feature in overlap_objects1]
    overlap_objects2_transformed = [(transform_bbox(center, frame2_warp_mat), key, feature) for center, key, feature in overlap_objects2]
    overlap_objects3_transformed = [(transform_bbox(center, frame3_warp_mat), key, feature) for center, key, feature in overlap_objects3]
    overlap_objects4_transformed = [(transform_bbox(center, frame4_warp_mat), key, feature) for center, key, feature in overlap_objects4]
    
    objects = overlap_objects1_transformed + overlap_objects2_transformed + overlap_objects3_transformed + overlap_objects4_transformed

    clustering = DBSCAN(eps=3, min_samples=2)

    data = np.array([obj[2] for obj in objects])
    clustering.fit(data)
    for label in set(clustering.labels_):
        if label == -1:  # noise
            continue
    for i, obj_label in enumerate(clustering.labels_):
        if obj_label == label:
            objects[i] = objects[i] + (global_id,)  # Append global_id to the object tuple
        global_id += 1
    #objects = overlap_objects1_transformed + overlap_objects2_transformed + overlap_objects3_transformed + overlap_objects4_transformed

    #cost_matrix = calculate_cost_matrix(overlap_objects1_transformed, overlap_objects2_transformed)
    #row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #matched_pairs = []
    #for r, c in zip(row_ind, col_ind):
        #if cost_matrix[r][c] <= max_distance:
            #matched_pairs.append((overlap_objects1_transformed[r][1], overlap_objects2_transformed[c][1]))
    return objects

def process_tracked_objects(tracked_objects, overlap_sec, warp_mat, id_dict, global_id, cam_id):
    overlap_objects = []
    for d in tracked_objects:
        xmin, ymin, xmax, ymax, id, feature,mean = d
        print(f"\n{feature}\n")
        center = calculate_center((xmin, ymin, xmax, ymax))
        key = f"{cam_id}_{id}"
        if is_inside(center, overlap_sec):
            overlap_objects.append((center, key, feature))
        elif key not in id_dict:
            id_dict[key] = global_id
            global_id += 1
    return overlap_objects, id_dict, global_id

cap1 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp")
cap2 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")
cap3 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
cap4 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp")
cap5 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.111/profile2/media.smp")
cap6 = VideoCaptureThreaded("rtsp://admin:saloris123!@192.168.0.115/profile2/media.smp")
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
tracker1 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker2 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker3 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker4 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker5 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
tracker6 = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
frame1_overlap_sec= np.loadtxt("frame1_overlap_section.txt").astype(np.float32)
frame2_overlap_sec = np.loadtxt("frame2_overlap_section.txt").astype(np.float32)
frame3_overlap_sec= np.loadtxt("frame3_overlap_section.txt").astype(np.float32)
frame4_overlap_sec = np.loadtxt("frame4_overlap_section.txt").astype(np.float32)
frame1_overlap_sec = frame1_overlap_sec.reshape((-1, 1, 2))
frame2_overlap_sec = frame2_overlap_sec.reshape((-1, 1, 2))
frame3_overlap_sec = frame3_overlap_sec.reshape((-1, 1, 2))
frame4_overlap_sec = frame4_overlap_sec.reshape((-1, 1, 2))
frame1_warp_mat = np.loadtxt("frame1_warpmatrix.txt")
frame2_warp_mat = np.loadtxt("frame2_warpmatrix.txt")
frame3_warp_mat = np.loadtxt("frame3_warpmatrix.txt")
frame4_warp_mat = np.loadtxt("frame4_warpmatrix.txt")
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
    num_frames += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
end_time = time.time()
processing_time = end_time - start_time
fps = num_frames / processing_time
print(f"Processing speed: {fps} frames per second")
cap1.stop()
cap2.stop()
cv2.destroyAllWindows()
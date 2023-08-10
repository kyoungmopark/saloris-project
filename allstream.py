from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import uvicorn
from multiprocessing import Process, Value
import cv2
import numpy as np
from DeepSORT_YOLOv5_Pytorch.deep_sort import DeepSort
import torch
from shapely.geometry import Point, Polygon
import time
import socket
import json

class Coordinate(BaseModel):
    x: float
    y: float

class ParkingSpot(BaseModel):
    cameraIndex: int
    parkingindex : int = 0
    coordinates: List[Coordinate]
    occupied: bool = False
    capture : int = 0
    vehicle_id: int = None

class Camera(BaseModel):
    rtsp: str

app = FastAPI()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
host = '127.0.0.1'
port = 4455
addr = (host, port)
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #일단 전부 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parking_lot = {}
camera_list = [
    "rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp",
    "rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp",
    "rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp",
    "rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp",
    "rtsp://admin:saloris123!@192.168.0.111/profile2/media.smp",
    "rtsp://admin:saloris123!@192.168.0.124/profile2/media.smp"
  ]

def point_in_polygon(x, y, polygon):
    point = Point(x, y)
    return polygon.contains(point)


async def get_stream_video(parking_lot,camera_index):
    cam = cv2.VideoCapture(camera_list[camera_index])
    tracker = DeepSort(r"C:\cctvproject\New\DeepSORT_YOLOv5_Pytorch\deep_sort\deep\checkpoint\ckpt.t7")
    while True:
        start_time = time.time()  
        ret, frame = cam.read()
        frame = cv2.resize(frame, (640, 480))
        pred = model(frame)
        boxes = pred.xyxy[0].cpu().numpy()[:, :4].astype(int) 
        class_ids = pred.xyxy[0].cpu().numpy()[:, 5].astype(int) 
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        tracked_objects = tracker.update(boxes, class_ids, frame)
        for d in tracked_objects:
                xmin, ymin, xmax, ymax, id,vx,vy = d
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                cv2.rectangle(frame, (xmin ,ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, str(id), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if camera_index in parking_lot:
                    for spot in parking_lot[camera_index]:
                        spot_coordinates = [(int(coord.x), int(coord.y)) for coord in spot.coordinates]
                        polygon = Polygon(spot_coordinates)
                        if point_in_polygon(center_x, center_y, polygon):
                            if spot.capture < 80:
                                spot.capture += 2
                                spot.vehicle_id = id
                                break     
        if camera_index in parking_lot:
            for spot in parking_lot[camera_index]:
                if spot.capture > 50:
                    if spot.occupied == False:
                        spot.occupied = True
                        data = json.dumps(spot.dict()).encode('utf-8')
                        client.sendto(data, addr)
                    spot.capture -= 1
                    if spot.capture == 50:
                        spot.occupied =False
                        spot.vehicle_id = None
                        spot.capture = 0
                        data = json.dumps(spot.dict()).encode('utf-8')
                        client.sendto(data, addr)


        for idx, spot in enumerate(parking_lot.get(camera_index, [])):
            xs = [int(coord.x) for coord in spot.coordinates]
            ys = [int(coord.y) for coord in spot.coordinates]
            lefts = sorted(spot.coordinates, key=lambda p: p.x)[:2]
            rights = sorted(spot.coordinates, key=lambda p: p.x)[2:]
            lefttop = (int(min(lefts, key=lambda p: p.y).x), int(min(lefts, key=lambda p: p.y).y))
            leftbottom = (int(max(lefts, key=lambda p: p.y).x), int(max(lefts, key=lambda p: p.y).y))
            righttop = (int(min(rights, key=lambda p: p.y).x), int(min(rights, key=lambda p: p.y).y))
            rightbottom = (int(max(rights, key=lambda p: p.y).x), int(max(rights, key=lambda p: p.y).y))
            cv2.line(frame, lefttop, righttop, (255, 0, 0), 2)
            cv2.line(frame, righttop, rightbottom, (255, 0, 0), 2)
            cv2.line(frame, rightbottom, leftbottom, (255, 0, 0), 2)
            cv2.line(frame, leftbottom, lefttop, (255, 0, 0), 2)
            cv2.putText(frame, str(spot.parkingindex), (lefttop[0] + 10, lefttop[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        f = buffer.tobytes()
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing speed: {processing_time} cameraindex : {camera_index}")
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        bytearray(f) + b'\r\n')
            

@app.post("/add_parking_spot")
def add_parking_spot(spot: ParkingSpot):
    camera_index = spot.cameraIndex
    coordinates = spot.coordinates
    if camera_index not in parking_lot:
        parking_lot[camera_index] = []
    spot.parkingindex = len(parking_lot[camera_index])
    parking_lot[camera_index].append(spot)
    return {"message": "Parking spot added successfully!"}

@app.get("/get_parking_info")
def get_parking_info():
    parking_info = []
    for camera_index, spots in parking_lot.items():
        for spot in spots:
            parking_info.append({
                "cameraIndex": camera_index,
                "parkingindex": spot.parkingindex,
                "occupied": spot.occupied,
                "capture": spot.capture,
                "vehicle_id": spot.vehicle_id,  
            })
    return {"parkingInfo": parking_info}

@app.post("/add_camera")
async def add_camera(camera: Camera):
    camera_list.append(camera.rtsp)
    return {"message": "Camera added successfully!"}    


@app.get("/get_cameras")
async def get_cameras():
    return {"cameras": camera_list}

@app.get("/video/{cam_id}")
async def main(cam_id: int):
    try:
        return StreamingResponse(get_stream_video(parking_lot,cam_id), media_type="multipart/x-mixed-replace; boundary=frame")
    except IndexError:
        raise HTTPException(status_code=404, detail="Camera not found")


def run_fastapi_server():
    uvicorn.run("allstream:app", host="localhost", port=8000, log_level="info")

if __name__ == "__main__":
    run_fastapi_server()
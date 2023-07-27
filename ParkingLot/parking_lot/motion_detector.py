import cv2
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE
import imutils
"""sort 코드 병합"""
from sort import * 
import torch 

class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame, model, mot_tracker):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.capture_list = []
        self.frame_list = []
        self.model = model 
        self.mot_tracker = mot_tracker

    def detect_motion(self):
        for video in self.video: 
            print(video)
            capture = cv2.VideoCapture(video)
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.capture_list.append(capture)
        print(self.capture_list)
        
        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = cv2.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = cv2.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=cv2.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while self.capture_list[0].isOpened():
            for capture in self.capture_list:
                result, frame = capture.read()
                self.frame_list.append(frame)
                if frame is None:
                    break
                if not result:
                    raise CaptureReadError("Error reading video capture on frame %s" % str(frame))
            
            frame = self.make_stitched_frame(self.frame_list)
            blurred = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_BLUE
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

            #sort 코드 병합
            preds = self.model(frame)
            detections = preds.pred[0].cpu().numpy()
            track_bbs_ids = self.mot_tracker.update(detections)
            for j in range(len(track_bbs_ids.tolist())):
        
                coords = track_bbs_ids.tolist()[j]
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                w = x2 - x1
                h = y2- y1
                name_idx = int(coords[4])
                name = "ID : {}".format(str(name_idx))
                colours = np.random.rand(3)
                cv2.rectangle(new_frame, (x1+w//2,y1+h//2), (x2+w//2,y2+h//2), colours, 2)
                cv2.putText(new_frame, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.9, colours, 2)
                cv2.imshow(str(self.video), new_frame)

            self.frame_list.clear()
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        cv2.destroyAllWindows()

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status
    
    def make_stitched_frame(self, frame_list): 
        frame_list[0] = cv2.resize(frame_list[0], dsize=(0,0), fx=0.25, fy=0.25)
        frame_list[1] = cv2.resize(frame_list[1], dsize=(0,0), fx=0.25, fy=0.25)
        frame_list[2] = cv2.resize(frame_list[2], dsize=(0,0), fx=0.25, fy=0.25)
        frame_list[3] = cv2.resize(frame_list[3], dsize=(0,0), fx=0.25, fy=0.25)

        frame_list[0] = frame_list[0][200:400, :]
        frame_list[1] = frame_list[1][200:400, :]
        frame_list[2] = frame_list[2][200:400, :]
        frame_list[3] = frame_list[3][200:400, :]

        frame_list[0] = imutils.rotate(frame_list[0], 180)
        frame_list[1] = imutils.rotate(frame_list[1], 180)
        hframe1 = cv2.hconcat([frame_list[0], frame_list[1]])
        hframe2 = cv2.hconcat([frame_list[2], frame_list[3]])
        stitched = cv2.vconcat([hframe1, hframe2])
        return stitched
    

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass

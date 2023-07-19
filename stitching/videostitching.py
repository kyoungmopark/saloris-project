import stitching
import cv2
import numpy as np

class VideoStitcher() :
    def __init__(self):
        setting = {"warper_type": "spherical", "detector": "orb", 
										"nfeatures" : 500, "try_use_gpu": True,}
        self.stitcher = stitching.Stitcher(**setting)

    def run_video(self): 
        # 실행 할 file 불러옴(camera setting 된 경우 cam으로 실행)
        cap1 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
        cap2 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")

        while cap1.isOpened():
            ret, frame1 = cap1.read()
            ret, frame2 = cap2.read()
            
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))

            frame_list = [frame1, frame2]
            stitched = self.stitcher.stitch(frame_list)
            #stitched = cv2.resize(stitched, (640,480))
            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            cv2.imshow('stitched', stitched)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__' :
    vs = VideoStitcher()
    vs.run_video()

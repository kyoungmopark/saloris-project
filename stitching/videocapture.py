import stitching
import cv2
import numpy as np

class VideoStitcher() :
    def __init__(self):
        setting = {"warper_type": "spherical", "detector": "orb", 
										"nfeatures" : 300, "try_use_gpu": True}
        self.stitcher = stitching.Stitcher(**setting)

    def run_video(self): 
        # 실행 할 file 불러옴(camera setting 된 경우 cam으로 실행)
        cap1 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp')
        cap2 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp')
        cap3 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp')


        while cap1.isOpened():
            ret, frame1 = cap1.read()
            ret, frame2 = cap2.read()
            ret, frame3 = cap3.read()


            #frame1 = cv2.resize(frame1, (640, 480))
            #frame2 = cv2.resize(frame2, (640, 480))
            #frame3 = cv2.resize(frame3, (640, 480))


            stitched = self.start_stitching(frame1, frame2, frame3)
            cv2.imshow('stitched', stitched)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # frame stitching 
    def start_stitching(self, frame1, frame2, frame3):

        f1_path = './data/f1.jpg'
        cv2.imwrite(f1_path, frame1)
        f2_path = './data/f2.jpg'
        cv2.imwrite(f2_path, frame2)
        f3_path = './data/f3.jpg'
        cv2.imwrite(f3_path, frame3)


        frame_list = [f1_path, f2_path, f3_path]
        stitched = self.stitcher.stitch(frame_list)

        return stitched


if __name__ == '__main__' :
    vs = VideoStitcher()
    vs.run_video()
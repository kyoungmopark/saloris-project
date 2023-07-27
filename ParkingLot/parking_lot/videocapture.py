import cv2
import numpy as np
import imutils 

class VideoStitcher() :
    def run_video(self): 
        # 실행 할 file 불러옴(camera setting 된 경우 cam으로 실행)
        #cap1 = cv2.VideoCapture('rtsp_front.mp4')
        cap1 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp')
        cap2 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp')
        cap3 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp')
        cap4 = cv2.VideoCapture('rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp')

        while cap1.isOpened():
            ret, frame1 = cap1.read()
            ret, frame2 = cap2.read()
            ret, frame3 = cap3.read()
            ret, frame4 = cap4.read() 
            frame1 = cv2.resize(frame1, dsize=(0,0), fx=0.25, fy=0.25)
            frame2 = cv2.resize(frame2, dsize=(0,0), fx=0.25, fy=0.25)
            frame3 = cv2.resize(frame3, dsize=(0,0), fx=0.25, fy=0.25)
            frame4 = cv2.resize(frame4, dsize=(0,0), fx=0.25, fy=0.25)
            #cv2.imshow('frame1', frame1)
            #cv2.imshow('frame2', frame2)
            #cv2.imshow('frame3', frame3)
            #cv2.imshow('frame4', frame4)
            #frame1 = imutils.rotate(frame1, 180)
            #frame2 = imutils.rotate(frame2, 180)
            hframe1 = cv2.hconcat([frame1, frame2])
            hframe2 = cv2.hconcat([frame3, frame4])
            stitched = cv2.vconcat([hframe1, hframe2])
            
            x,y,w,h = cv2.selectROI('stitched', stitched, False)
            if w and h:
                roi = stitched[y:y+h, x:x+w]
                cv2.rectangle(stitched, pt1=(x,y), pt2=(x+w, y+h), color=(0,255,0), thickness=5)
            cv2.imshow('stitched', stitched)
            
            #f1_path = './images/f1.jpg'
            #cv2.imwrite(f1_path, frame1)
            #f2_path = './images/f2.jpg'
            #cv2.imwrite(f2_path, frame2)
            #f3_path = './images/f3.jpg'
            #cv2.imwrite(f3_path, frame3)
            #f4_path = './images/f4.jpg'
            #cv2.imwrite(f4_path, frame4)
            #result_path = './images/result.jpg'
            #cv2.imwrite(result_path, stitched)
            
            if cv2.waitKey(1) == ord('q'):
                print("image capture done!")
                break

if __name__ == '__main__' :
    vs = VideoStitcher()
    vs.run_video()
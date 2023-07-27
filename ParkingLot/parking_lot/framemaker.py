import cv2
import numpy as np

def make_stitched_frame(frame_list): 
    for frame in frame_list:
        frame = cv2.resize(frame, dsize=(0,0), fx=0.25, fy=0.25)

    #frame1 = frame1[200:400, 50:600]
    #frame2 = frame2[200:400, 150:700]
    #frame3 = frame3[190:390, 100:650]
    #frame4 = frame4[200:400, 150:700]

    hframe1 = cv2.hconcat([frame_list[0], frame_list[1]])
    hframe2 = cv2.hconcat([frame_list[2], frame_list[3]])
    stitched = cv2.vconcat([hframe1, hframe2])
    
    return stitched

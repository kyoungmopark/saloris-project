from sort import * 
import torch 
import cv2 
import numpy as np 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='pinkcar.pt', force_reload=True)
model.float()
model.eval() 
vid1 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")
vid2 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp")
vid3 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp")
vid4 = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.107/profile2/media.smp")
#create instance of SORT
mot_tracker = Sort()
while(True):

    ret, frame1 = vid1.read()
    ret, frame2 = vid2.read() 
    ret, frame3 = vid3.read()
    ret, frame4 = vid4.read()
     
    frame1 = cv2.resize(frame1, dsize=(0,0), fx=0.25, fy=0.25)
    frame2 = cv2.resize(frame2, dsize=(0,0), fx=0.25, fy=0.25)
    frame3 = cv2.resize(frame3, dsize=(0,0), fx=0.25, fy=0.25)
    frame4 = cv2.resize(frame4, dsize=(0,0), fx=0.25, fy=0.25)

    hframe1 = cv2.hconcat([frame1, frame2])
    hframe2 = cv2.hconcat([frame3, frame4])
    image_show = cv2.vconcat([hframe1, hframe2])
    #image_show = cv2.resize(image_show, dsize=(0,0), fx=0.3, fy=0.3)
    preds = model(image_show)
    detections = preds.pred[0].cpu().numpy()
    track_bbs_ids = mot_tracker.update(detections)

    for j in range(len(track_bbs_ids.tolist())):
        
        coords = track_bbs_ids.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = "ID : {}".format(str(name_idx))
        #color = colours[name_idx]
        cv2.rectangle(image_show, (x1,y1), (x2,y2), 2)
        cv2.putText(image_show, name, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, 2)
        cv2.imshow('Image', image_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#After the loop release the cap object 
vid1.release()
vid2.release()
vid3.release()
vid4.release() 
#Destroy all the windows 
cv2.destroyAllWindows()


import cv2
import sys
import numpy as np

url1 = 'rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp'
url2 = 'rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp'
url3 = 'rtsp://admin:saloris123!@192.168.0.106/profile2/media.smp'

cap1 = cv2.VideoCapture(url1)
cap2 = cv2.VideoCapture(url2)
cap3 = cv2.VideoCapture(url3)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    # Read frame from each camera
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    timestamp1 = cap1.get(cv2.CAP_PROP_POS_MSEC)
    timestamp2 = cap2.get(cv2.CAP_PROP_POS_MSEC)
    timestamp3 = cap3.get(cv2.CAP_PROP_POS_MSEC)
    frame1 = cv2.resize(frame1, None, fx=0.3, fy=0.3)
    frame2 = cv2.resize(frame2, None, fx=0.3, fy=0.3)
    frame3 = cv2.resize(frame3, None, fx=0.3, fy=0.3)
    img_size = (frame1.shape[1], frame1.shape[0])
    if not ret1 or not ret2 or not ret3:  # Check if frames were successfully read
        break
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    matches = bf.match(des1,des2)
    
    if len(matches)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        h,w = frame1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        frame2 = cv2.polylines(frame2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        # Transform frame2 using homography matrix M
        frame2_transformed = cv2.warpPerspective(frame2, M, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))

        # Now paste them together
        frame2_transformed[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
        cv2.imshow('Stitched', frame2_transformed)
        
        # Draw matches
        matched = cv2.drawMatches(frame1, kp1, frame2_transformed, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', matched)

    else:
        print( "Not enough matches are found - %d/%d" % (len(matches),10))

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    #cv2.putText(frame1, str(timestamp1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.putText(frame2, str(timestamp2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #cv2.putText(frame3, str(timestamp3), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #frame1_matrix = np.loadtxt('frame1.txt')
    #frame2_matrix = np.loadtxt('frame2.txt')
    #frame3_matrix = np.loadtxt('frame3.txt')
    #frame1 = cv2.warpPerspective(frame1, frame1_matrix, img_size, flags=cv2.INTER_LINEAR)
    #frame2 = cv2.warpPerspective(frame2, frame2_matrix, img_size, flags=cv2.INTER_LINEAR)
    #frame3 = cv2.warpPerspective(frame3, frame3_matrix, img_size, flags=cv2.INTER_LINEAR)
    #frame1 = frame1[:, :frame1.shape[1]//2]
    #frame2 = frame2[:, :frame2.shape[1]//2]
    #frame3 = frame3[:, :frame3.shape[1]//2]
    #frames = [frame1, frame2, frame3]
    #stitcher = cv2.Stitcher_create()
    #status, stitched = stitcher.stitch(frames)
    #if status == cv2.Stitcher_OK:
     #   cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
     #   cv2.imshow('Stitched Image', stitched)
    #else:
     #   print('Error during stitching, status code = %d' % status)
      #  sys.exit()

    #cv2.imshow("Camera 3", frame3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()      
cv2.destroyAllWindows()
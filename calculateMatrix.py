import cv2
import numpy as np

win_name = 'camera1'
win_name2 = 'Canvas2'
url1 = 'rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp'
cap1 = cv2.VideoCapture(url1)
ret1, img = cap1.read()
img = cv2.resize(img, None, fx=0.3, fy=0.3)
rows, cols = img.shape[:2]
exec
canvas2 = np.zeros((500, 500, 3), np.uint8)
draw = img.copy()
pts_cnt1 = 0
pts_cnt2 = 0
pts1 = np.zeros((4, 2), dtype=np.float32)
pts2 = np.zeros((4, 2), dtype=np.float32)

def onMouse1(event, x, y, flags, param):     # 마우스 이벤트 콜백 함수 구현

    global pts_cnt1, pts1,pts2,pts_cnt2                       # 마우스로 찍은 좌표의 개수 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)   # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)
        pts1[pts_cnt1] = [x, y]       # 마우스 좌표 저장
        pts_cnt1 += 1
        print(pts_cnt1)
        print(pts_cnt2)
        if pts_cnt1 == 4 and pts_cnt2 == 4:             # 좌표 4개 수집
            # 좌표 4개 중 상하좌우 찾기
            sm1 = pts1.sum(axis=1)                # 4쌍 좌표 각각 x+y 계산
            diff1 = np.diff(pts1, axis=1)         # 4쌍 좌표 각각 x-y 계산
            topLeft1 = pts1[np.argmin(sm1)]        # x+y가 가장 작은 값이 좌상단 좌표
            bottomRight1 = pts1[np.argmax(sm1)]    # x+y가 가장 큰 값이 우하단 좌표
            topRight1 = pts1[np.argmin(diff1)]     # x-y가 가장 작은 값이 우상단 좌표
            bottomLeft1 = pts1[np.argmax(diff1)]   # x-y가 가장 큰 값이 좌하단 좌표S
            # 변환 전 4개의 좌표
            pts1 = np.float32([topLeft1, topRight1, bottomRight1, bottomLeft1])
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            np.savetxt('frame3_overlap_section.txt', pts1)
            np.savetxt('frame3_warpmatrix.txt', mtrx)
            print("완료\n")

def onMouse2(event, x, y, flags, param):     # 마우스 이벤트 콜백 함수 구현

    global pts2,pts_cnt2                      # 마우스로 찍은 좌표의 개수 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(canvas2, (x, y), 10, (0, 255, 0), -1)   # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name2, canvas2)
        pts2[pts_cnt2] = [x, y]       # 마우스 좌표 저장
        pts_cnt2 += 1

cv2.imshow(win_name, img)
cv2.imshow(win_name2, canvas2)
cv2.setMouseCallback(win_name, onMouse1) # 마우스 콜백 함수를 GUI 윈도에 등록
cv2.setMouseCallback(win_name2, onMouse2)
cv2.waitKey(0)
cv2.destroyAllWindows()

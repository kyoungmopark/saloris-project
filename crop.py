# import cv2
# import numpy as np
#
# win_name = "scanning"
# stream_url = "rtsp://admin:saloris123!@192.168.0.121/profile2/media.smp"
# cap = cv2.VideoCapture(stream_url)
# ret, img = cap.read()
# cap.release()
# img = cv2.resize(img, None, fx=0.45, fy=0.45)
# rows, cols = img.shape[:2]
# draw = img.copy()
# pts_cnt = 0
# pts = np.zeros((4, 2), dtype=np.float32)
#
# def onMouse(event, x, y, flags, param):
#     global pts_cnt
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # 좌표에 초록색 동그라미 표시
#         cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
#         cv2.imshow(win_name, draw)
#
#         # 마우스 좌표 저장
#         pts[pts_cnt] = [x, y]
#         pts_cnt += 1
#         if pts_cnt == 4:
#             # 좌표 4개 중 상하좌우 찾기
#             sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
#             diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산
#
#             topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
#             bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
#             topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
#             bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표
#
#             # 변환 전 4개 좌표
#             pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
#
#             # 변환 후 영상에 사용할 서류의 폭과 높이 계산
#             w1 = abs(bottomRight[0] - bottomLeft[0])
#             w2 = abs(topRight[0] - topLeft[0])
#             h1 = abs(topRight[1] - bottomRight[1])
#             h2 = abs(topLeft[1] - bottomLeft[1])
#             width = int(max([w1, w2]))  # 폭의 최대값을 정수로 변환
#             height = int(max([h1, h2]))  # 높이의 최대값을 정수로 변환
#
#             # 변환 후 4개 좌표
#             pts2 = np.float32([[0, 0], [width - 1, 0],
#                                [width - 1, height - 1], [0, height - 1]])
#
#             # 변환 행렬 계산
#             mtrx = cv2.getPerspectiveTransform(pts1, pts2)
#             # 원근 변환 적용
#             result = cv2.warpPerspective(img, mtrx, (width, height))
#             cv2.imshow('scanned', result)
#             cv2.imwrite('real_img/longimg2.png', result)
#
# cv2.imshow(win_name, img)
#
# cv2.setMouseCallback(win_name, onMouse)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

win_name = "scanning"
cap = cv2.VideoCapture("rtsp://admin:saloris123!@192.168.0.114/profile2/media.smp")


ret, img = cap.read()
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
mtrx = None  # 변환 행렬을 전역 변수로 선언

def onMouse(event, x, y, flags, param):
    global pts_cnt, mtrx  # mtrx를 전역 변수로 사용
    if event == cv2.EVENT_LBUTTONDOWN:
        # 좌표에 초록색 동그라미 표시
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow(win_name, draw)

        # 마우스 좌표 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        if pts_cnt == 4:
            # 좌표 4개 중 상하좌우 찾기
            sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = int(max([w1, w2]))  # 폭의 최대값을 정수로 변환
            height = int(max([h1, h2]))  # 높이의 최대값을 정수로 변환

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            print(mtrx)


cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, onMouse)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to receive frame from webcam.")
        break

    # 변환 행렬이 존재하면 원근 변환 적용
    if mtrx is not None:
        height, width = img.shape[:2]
        warped_frame = cv2.warpPerspective(frame, mtrx, (width, height))
        cv2.imshow('scanned', warped_frame)

    cv2.imshow(win_name, frame)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

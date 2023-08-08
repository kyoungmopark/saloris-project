import cv2

def get_stream_video(cam_id, width, height):
    cap = cv2.VideoCapture(cam_id)
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, (width, height))  # resize with given parameters
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')
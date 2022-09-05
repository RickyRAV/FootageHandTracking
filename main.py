import os
import sys
import time
import mediapipe as mp
import cv2
footage_path = os.path.abspath('hand.mp4')
capture = cv2.VideoCapture(footage_path)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
prevFrameTime = 0
newFrameTime = 0
with mp_hands.Hands(min_tracking_confidence=0.8, min_detection_confidence=0.2, max_num_hands=2) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        gray = cv2.resize(frame, (430, 720))
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(gray, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 0, 0), circle_radius=2, thickness=5))

        font = cv2.FONT_HERSHEY_SIMPLEX
        newFrameTime = time.time()
        fps = 1 / (newFrameTime - prevFrameTime)
        prevFrameTime = newFrameTime
        fps = int(fps)
        fps = 'FPS:' + str(fps)
        cv2.putText(gray, fps, (7, 40), font, 1.3, (165, 100, 100), 2, cv2.LINE_AA)
        cv2.imshow('Video Reference Tracking', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()

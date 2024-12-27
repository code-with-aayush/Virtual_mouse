import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

pyautogui.FAILSAFE = False

wCam, hCam = 600, 480
frameR = 100
smoothening = 6

# Adjustments for the rectangle
left_adjust = 60      # Shift rectangle to the right by 60 pixels
right_adjust = 8      # Increase rectangle width by 8 pixels
bottom_adjust = 20    # Decrease height by 20 pixels

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

index_y = 0
index_x = 0

x1, y1 = 0, 0
x2, y2 = 0, 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    top_left = (frameR + left_adjust, frameR)  
    bottom_right = (wCam - frameR + right_adjust, hCam - frameR - bottom_adjust)  

    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)


    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * (frame_width))
                y = int(landmark.y * (frame_height))

                if id == 8:
                    cv2.circle(frame, (x, y), 10, (0, 0, 0))

                    # Map the hand cursor position to the screen
                    x3 = np.interp(x, (frameR + left_adjust, wCam - frameR + right_adjust), (0, screen_width))
                    y3 = np.interp(y, (frameR, hCam - frameR - bottom_adjust), (0, screen_height))

                    try:
                        pyautogui.moveTo(x3, y3)
                    except Exception as e:
                        pass

                if id == 6:
                    x1, y1 = int(x), int(y)

                if id == 4:
                    x2, y2 = int(x), int(y)

                
                    cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x2, y2), (x1, y1), (0, 0, 255), 3)
                    cv2.circle(frame, ((x2 + x1) // 2, (y2 + y1) // 2), 10, (0, 0, 255), cv2.FILLED)

                    length = int(math.hypot(x2 - x1, y2 - y1))
                    print(length)

                    if length < 26:
                        cv2.line(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)
                        cv2.circle(frame, ((x2 + x1) // 2, (y2 + y1) // 2), 10, (0, 255, 0), cv2.FILLED)
                        cv2.circle(frame, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
                        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()

    
    cv2.imshow("frame", frame)  

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

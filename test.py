import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural movement
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Example: check if hand is open or closed using landmarks
            thumb_tip = hand_landmarks.landmark[4].y
            index_tip = hand_landmarks.landmark[8].y
            middle_tip = hand_landmarks.landmark[12].y

            if index_tip < hand_landmarks.landmark[6].y:  # Finger raised
                print("Next Slide")
                pyautogui.press("right")  # Move to next PPT slide

            elif index_tip > hand_landmarks.landmark[6].y and middle_tip > hand_landmarks.landmark[10].y:
                print("Previous Slide")
                pyautogui.press("left")  # Move to previous PPT slide

    cv2.imshow("Hand Tracking PPT Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

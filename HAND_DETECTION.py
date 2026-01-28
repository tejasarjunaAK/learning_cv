import cv2
import mediapipe as mp
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.7)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cv2.read()
    if not ret:
        break
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)#mediapipe requires rgb and opencv gives output as bgr
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw(frame,landmarks,mp_hands.HAND_CONNECTIONS)
    cv2.imshow(frame)
    if cv2.waitKey(1)& 0xFF=="q":
        break
cap.release()
cap.destroyAllWindows()
            
    
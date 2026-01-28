import cv2
import numpy as np
import matplotlib.pyplot as plt 

#There is no difference between video files and webcam
cap=cv2.VideoCapture(0)
cap.set(3,640) #width
cap.set(4,480)# height 3: cv2.CAP_PROP_FRAME_HEIGHT
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("camera",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
    
    


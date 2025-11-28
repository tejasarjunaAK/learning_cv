import cv2
#to read video
vid= cv2.VideoCapture('videofile.api')

# Toget each frame of video
ret =True
while ret:
    ret,frame=vid.read()
    
    #if we want to view frames
    cv2.imshow('frame',frame)
    cv2.waitKey(40)# we will give number and 40 millisecond if we give 0 we will have to press the keyboard many times
    # each frame is numpy array like image. It is same as image at particular part of video
    
    


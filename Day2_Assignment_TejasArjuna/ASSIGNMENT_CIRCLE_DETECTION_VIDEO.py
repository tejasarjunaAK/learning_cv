import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 
class video_circle():
    def __init_(self,frame):
      self.img=frame
      
    def circle_detect(self,param2):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(21,21),0)
        circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,minDist=50,param1=100,param2=param2,minRadius=10,maxRadius=0)
        return circles
        
    def canny_identification(self,param2):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(21,21),0)
        canny=cv2.Canny(blurred,40,120)
        kernel=np.ones(self.img.shape)
       
        dilated=cv2.dilate(canny,kernel,iterations=1)
        circles=self.circle_detect(param2)
        circles=np.around(circles).astype(int)
        if circles is None:
            return 0,0
        if circles is not None:
            penaltyB=0
            penaltyA=0
            count=0
            for x,y,r in circles[0]:
                cv2.circle(canny,(x,y),r,(0,0,255),2)
                count+=1
            diff_score=cv2.absdiff(dilated,canny).mean()
            penaltyB=diff_score/255
            if count>12:
                penaltyA=count/100
            return penaltyA,penaltyB
    def compute(self):
        score_max=-1e8
        param2=0
        for i in range(50,80,2):
            circles=self.circle_detect(i)
            penaltyA,penaltyB=self.canny_identification(circles)
            score=-penaltyA-penaltyB
            if score>score_max:
                score_max=score
                param2=i
        print(f"Best parameter is {param2}")
    def combined(self):
        
        
def frame_get(video_file):
    if os.path.exists(video_file)==False:
        raise FileNotFoundError("File not present")
    cap=cv2.VideoCapture(video_file)
    
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out=cv2.VideoWriter("circle_detector",cv2.VideoWriter_fourcc(*"mpv4"),fps,(width,height))
    while True:
        ret,frame=cap.read()
        while not ret:
            break
        detector=video_circle(frame)
        new_frame=detector.combined()
        out.write(new_frame)
        
def main():
    video_file=input("Enter video file path")
    frame_get(video_file)
main()   
        
        
    
            
            


        
        
        
        
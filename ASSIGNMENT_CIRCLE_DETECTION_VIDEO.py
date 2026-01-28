import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 
class video_circle():
    def __init__(self,frame):
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
        kernel=np.ones((5,5),dtype=np.uint8)
       
        dilated=cv2.dilate(canny,kernel,iterations=1)
        circles=self.circle_detect(param2)
        if circles is None:
            return 1,1
        circles=np.around(circles).astype(int)
        if circles is not None:
            penaltyB=0
            penaltyA=0
            count=0
            for x,y,r in circles[0]:
                cv2.circle(canny,(x,y),r,255,2)
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
            penaltyA,penaltyB=self.canny_identification(i)
            score=-penaltyA-penaltyB
            if score>score_max:
                score_max=score
                param2=i
        return param2
    def combined(self):
        frame=self.img.copy()
        param2=self.compute()
        circles=self.circle_detect(param2)
        if circles is None:
            return frame
        circles = np.uint16(np.around(circles[0])).astype(int)
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        for (cx, cy, r) in circles:
        # ROI around circle
          pad = int(r * 1.2)
          x1, y1 = max(cx - pad, 0), max(cy - pad, 0)
          x2, y2 = min(cx + pad, self.img.shape[1]-1), min(cy + pad, self.img.shape[0]-1)

          roi = gray[y1:y2, x1:x2]

        # Edge / binary mask
          edges = cv2.Canny(roi, 50, 150)
          contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          if not contours:
             continue

        # Pick contour whose centroid is closest to Hough center
          best_cnt = None
          best_dist = 1e9
          for cnt in contours:
            area_min=30
            area = cv2.contourArea(cnt)
            if area < area_min:    # ignore tiny noisy contours
               continue
            (mx, my), _ = cv2.minEnclosingCircle(cnt)
            # map back to full-image coords
            gx, gy = x1 + mx, y1 + my
            d = (gx - cx)**2 + (gy - cy)**2
            if d < best_dist:
                best_dist = d
                best_cnt = cnt

          if best_cnt is None:
            continue

        # Fit circle to best contour
          (fx, fy), fr = cv2.minEnclosingCircle(best_cnt)
          fx, fy = int(x1 + fx), int(y1 + fy)
          fr = int(fr)
          cv2.circle(frame, (fx, fy), fr, (0, 0, 255), 2)
        return frame  
def frame_get(video_file):
    if os.path.exists(video_file)==False:
        raise FileNotFoundError("File not present")
    cap=cv2.VideoCapture(video_file)
    
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out=cv2.VideoWriter("circle_detector.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps,(int(width),int(height)))
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        detector=video_circle(frame)
        new_frame=detector.combined()
        out.write(new_frame)
    cap.release()
    out.release()

        
def main():
    video_file=input("Enter video file path")
    frame_get(video_file)
main()   
        
        
    
            
            


        
        
        
        
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
class autotune_my_method():
    def __init__(self,image_path):
        if os.path.exists(image_path)==False:
            raise FileNotFoundError
        if not os.path.splitext(image_path)[1].lower() in ('.png','.jpeg','.jpg','.bmp'):
            raise ValueError("File not with appropriate extensions")
        self.img=cv2.imread(image_path)
    def circle_detection(self,param2):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(21,21),0)
        circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,minDist=50,param1=100,param2=param2,minRadius=10,maxRadius=0)
        return circles
    def canny_identification(self,circles):
        copy=self.img.copy()
        canny2=cv2.Canny(copy,50,150)
        kernel=np.ones((5,5),dtype=np.uint8)
        dilated=cv2.dilate(canny2,kernel,iterations=1)
        count=0
        if circles is not None:
           circles=np.around(circles)
           for x,y,r in circles[0]:
               count+=1
               cv2.circle(canny2,(int(x),int(y)),int(r),(255),2)
               
        diff_score=cv2.absdiff(dilated,canny2).mean()
        penaltyA=0
        penaltyB=0
        penaltyB=diff_score/(255)
        if count>10:
            penaltyA=count/100
        return copy,penaltyA,penaltyB
    def param_selection(self):
        refined=self.img.copy()
        copy=self.img.copy()
        score_max=-1e8
        par_max=50
        for i in range(50,80,2):
            circles=self.circle_detection(i)
            image,penaltyA,penaltyB=self.canny_identification(circles)
            score=-penaltyA-penaltyB
            if score>score_max:
                score_max=score
                par_max=i
        circles=self.circle_detection(par_max)
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
          cv2.circle(refined, (fx, fy), fr, (0, 0, 255), 2)

        print(f"Best parameter is {par_max}")
        
        plt.figure(figsize=(10,5))
        plt.subplot(2,1,1)
        plt.imshow(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))
        plt.title("Original image")
        plt.axis("off")
        plt.subplot(2,2,1)
        plt.imshow(cv2.cvtColor(refined,cv2.COLOR_BGR2RGB))
        plt.title("Image_with_Circles")
        plt.axis("off")
        plt.show()
def main():
    image_path=input("Enter file path")
    detector=autotune_my_method(image_path)
    param_selection=detector.param_selection()
main()
                
            
               
            
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
class line_circle_detector():
    def __init__(self, image_path):
        if os.path.exists(image_path)==False:
            raise FileNotFoundError(f"The image path {image_path} does not exist.")
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.blurred=cv2.GaussianBlur(self.gray,(7,7),0)
        self.canny=cv2.Canny(self.blurred,50,150)
        self.copy=self.image.copy()
    def line_detecting(self):
        self.lines=cv2.HoughLinesP(self.canny,1,np.pi/180,threshold=90,minLineLength=50,maxLineGap=10)
        if not self.lines is None:
            for line in self.lines:
               x1,y1,x2,y2=line[0]
               cv2.line(self.copy,(x1,y1),(x2,y2),(0,255,255),2)
    def circle_detecting(self):
        circles=cv2.HoughCircles(self.gray,cv2.HOUGH_GRADIENT,dp=0.9,minDist=100,minRadius=50,maxRadius=0,param1=120,param2=80)
        if circles is not None:
            circles=np.uint16(np.around(circles))
            for circle in circles[0]:
                cx,cy,radius=circle
                cv2.circle(self.copy,(cx,cy),radius,color=(255,255,0),thickness=2)
    def show_image(self):
        plt.figure(figsize=(10,8))
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(self.copy,cv2.COLOR_BGR2RGB))
        plt.title("Image with lines and circles")
        plt.axis("off")
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        plt.show()
image_path=input("Enter path of image")
detector=line_circle_detector(image_path)
detector.line_detecting()
detector.circle_detecting()
detector.show_image()
                
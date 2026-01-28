import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class Circle_Detector:
    def __init__(self,image_path):
        # Initialising global variables of class
        self.image_path = image_path
        extensions=('.png','.jpg','.jpeg','.bmp')
        k=os.path.splitext(self.image_path)
        if k[1].lower() not in extensions: # Error handling for invalid image format
            raise ValueError("Invalid image format")  
        if not os.path.exists(self.image_path): # Error handling if file not found
            raise FileNotFoundError("File not found")    
        self.original_image = cv2.imread(self.image_path)
    def hough_circle(self):
        #making blurred gray image and detecting circles using HoughCircles
        gray=cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(21,21),0)
        return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,param1=100, param2=70, minRadius=20, maxRadius=100)
    #highe parameter lower circles,
    def draw_circles(self,circles):
        #draws circles on the image copy using the circles detected earlier
        output_image = self.original_image.copy()
        
        radius=[]
        center=[]
        if circles is not None:
            circles = np.uint16(np.around(circles))# np.around(array) will round of values to nearest integer
            for i in range(circles.shape[1]):
                
                # Draw the outer circle
                center.append((circles[0][i][0],circles[0][i][1]))
                radius.append(circles[0][i][2])
                cv2.circle(output_image, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 255, 0), 2)
                cv2.putText(output_image,f'{radius[i]}',(circles[0][i][0],circles[0][i][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                cv2.circle(output_image,(circles[0][i][0],circles[0][i][1]),3,(0,0,255),-1)
        return output_image,radius,center
def main():
    img_path =input("Enter the path of image:")
    detector=Circle_Detector(img_path)   
    circles=detector.hough_circle()
    
    output_image,radius,center=detector.draw_circles(circles)
    
    if len(radius)==0:
        print("no circles found")
    else:
      arr_radius=np.array(radius)
      #writing statistics of circles into file
    
      with open("output_statistics.txt",'w') as f:
        f.write(f"Number of circles detected: {arr_radius.shape[0]} \n")
        f.write(f"Minimum radius of circles is :{arr_radius.min()} \n")
        f.write(f"Maximum radius of circles is {arr_radius.max()} \n")
        f.write(f"Average radius of circles is {arr_radius.mean()} \n")
        f.write(f"center \t radius \n")
        for i in range(arr_radius.shape[0]):
            f.write(f"{center[i]} \t {arr_radius[i]} \n")

    cv2.imwrite("circles_detected_image.png",output_image)
    plt.figure(figsize=(10,5))
    plt.imshow(cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles")
    plt.axis("off")
    plt.show()
   
main()
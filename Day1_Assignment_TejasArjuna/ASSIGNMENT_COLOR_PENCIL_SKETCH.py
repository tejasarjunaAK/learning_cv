import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class color_sketch():
    def __init__(self,image_path):
        
        if os.path.exists(image_path)==False:
            raise FileNotFoundError("The specified image path does not exist.")
        if os.path.splitext(image_path)[1].lower() not in ['.png','.jpeg','.jpg','.bmp']:
            raise ValueError("The speg_pathcified file is not a valid image format.")
        self.img=cv2.imread(image_path)
    def pencil_sketch_algorithm(self):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        invert=255-gray
        blurred=cv2.GaussianBlur(invert,(21,21),0)
        inverted_blurred=255-blurred
        sketch=cv2.divide(gray,inverted_blurred,scale=256.0)
        blurred_gray=cv2.GaussianBlur(gray,(21,21),0)
        canny2 = cv2.Canny(blurred_gray, 30,100)   #smaller the min threshold more edges detected
        conditional_canny2= canny2.copy()
        conditional_canny2[blurred_gray>180]=255-canny2[blurred_gray>180] #inverting canny where image is bright(it will make image whitish)
        sketch2 = cv2.addWeighted(sketch, 0.8,conditional_canny2, 0.3, 0) 
        return sketch2.astype(np.uint8)
    def color_sketch_algorithm(self):
        value=self.pencil_sketch_algorithm()
        hsv=cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(
        hsv[:,:,1].astype(np.float32) * 0.9,0,255).astype(np.uint8)
        hsv[:,:,2]=value
        color_sketch=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return color_sketch.astype(np.uint8)
def main():
    image_path=input("Enter the path of the image: ")
    sketcher=color_sketch(image_path)
    color_sketch_img=sketcher.color_sketch_algorithm()
    cv2.imwrite("color_sketch_image.png",color_sketch_img)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,2)
    plt.imshow(color_sketch_img)
    plt.title("Color Sketch Image")
    plt.axis("off")
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(sketcher.img,cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.show()
              
main()
                              
        
                            
import cv2
import numpy as np
import os

class pencil_sketch:
    def __init__(self,image_path):
        #initialising global variables of class
        self.image_path = image_path
        extensions=('.png','.jpg','.jpeg','.bmp')
        k=os.path.splitext(self.image_path)
        if k[1].lower() not in extensions:#error handling for invalid image format
            raise ValueError("Invalid image format")  
        if not os.path.exists(self.image_path):#error handling if file not found
            raise FileNotFoundError("File not found")    
        self.original_image = cv2.imread(self.image_path)
    def inversion(self,img):
        # for inverting image -negative of image
        return 255-img
    def img_division(self,img1,img2):
        # blending image by division and clipping
        return cv2.divide(img1,img2,scale=256)
    def combined_show(self,img1,img2):
        # to show images side by side
        combined = np.hstack((img1, img2))
        cv2.imshow('Combined Image', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def compute(self):
        #total algorithm for pencil sketch
        gray=cv2.cvtColor(self.original_image,cv2.COLOR_BGR2GRAY)
        inverted_image = self.inversion(gray)
        blurred = cv2.GaussianBlur(inverted_image, (21,21),0)
        inverted_blurred = self.inversion(blurred)
        pencil_sketch_image = self.img_division(gray,inverted_blurred)
        cv2.imwrite('pencil_sketch.png',pencil_sketch_image)
        #till this is the pencil sketch using normal method
        #Now we use canny for edge detection(as in some images edges were not viewable as outlined)
        blurred_gray=cv2.GaussianBlur(gray,(21,21),0)
        canny2 = cv2.Canny(blurred_gray, 30,100)   #smaller the min threshold more edges detected
        conditional_canny2=conditional_canny = canny2.copy()
        conditional_canny2[blurred_gray>180]=255-canny2[blurred_gray>180] #inverting canny where image is bright(it will make image whitish)
        sketch2 = cv2.addWeighted(pencil_sketch_image, 0.8,conditional_canny2, 0.3, 0) 
        
        sketch2=np.clip(sketch2,0,255).astype(np.uint8)
        cv2.imwrite("pencil_canny.png",sketch2)
        sketch2=cv2.cvtColor(sketch2,cv2.COLOR_GRAY2BGR)
        
        self.combined_show(self.original_image,sketch2)

       




        
def main():
    img_path =input("Enter the path of image:")
    pencil=pencil_sketch(img_path)   
    pencil.compute()
    
main()
    


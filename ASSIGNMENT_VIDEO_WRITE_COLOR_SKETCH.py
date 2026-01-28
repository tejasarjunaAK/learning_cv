import cv2
import numpy as np
import os
import tqdm
class Video_Sketch():
    def __init__(self,video_path):
        if os.path.exists(video_path)==False:
            raise FileNotFoundError("Videofile not found")
        self.cap=cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("video not opening")
    def get_info(self):
        height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps=int( self.cap.get(cv2.CAP_PROP_FPS))
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        out=cv2.VideoWriter("color_sketched.mp4",fourcc,fps,(width,height))
    def pencil_sketch_algorithm(self,img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        invert=255-gray
        blurred=cv2.GaussianBlur(invert,(21,21),0)
        invert_blurred=255-blurred
        pencil_sketch=cv2.divide(gray,invert_blurred,scale=256)
        blurred_gray=cv2.GaussianBlur(gray,(21,21),0)
        canny2 = cv2.Canny(blurred_gray, 30,100)   #smaller the min threshold more edges detected
        conditional_canny2 = canny2.copy()
        conditional_canny2[blurred_gray>180]=255-canny2[blurred_gray>180] #inverting canny where image is bright(it will make image whitish)
        sketch2 = cv2.addWeighted(pencil_sketch, 0.8,conditional_canny2, 0.3, 0) 
        return pencil_sketch
    
    def color_sketch_algorithm(self,img):
        value=self.pencil_sketch_algorithm(img)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv[:,:,2]=value
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.9,0,255).astype(np.uint8)
        img= cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return img
        
        
    def video_write(self):
        while True:
            ret,frame=self.cap.read()
            if not ret:
                break
            final_frame=self.color_sketch_algorithm(frame)
            out=self.get_info()
            out.write(final_frame)
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
    

            
            
            
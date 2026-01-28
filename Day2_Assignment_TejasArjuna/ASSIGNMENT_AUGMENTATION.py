import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class augmentation:
    def __init__(self,image_path):
        self.image_path=image_path
        if os.path.exists(image_path)==False:
            raise FileNotFoundError("File not found")
        if os.path.splitext(image_path)[1].lower() not in ('.png','.jpg','.jpeg','.bmp'):
            raise ValueError("File not of appropriate type")
        
        
        self.img=cv2.imread(self.image_path)
    def augmentation(self):
        # brightness augmentation
            bright_img=self.img.astype(np.int16)
            brightness_factor=30
            bright_img=np.clip(bright_img+brightness_factor,0,255).astype(np.uint8)
            cv2.imwrite("brightness_augmented_image.jpg",bright_img)
        # contrast augmentation
            contrast_factor=1.5
            contrast_img=self.img.astype(np.float32)
            contrast_img=np.clip((contrast_img-128)*contrast_factor+128,0,255).astype(np.uint8)
            cv2.imwrite("contrast_augmented_image.jpg",contrast_img)
        # rotation augmentation - random angle between -40 and +40
            angle=np.random.uniform(-40,40)
            height,width=self.img.shape[:2]
            center=(width//2,height//2)
            rotation_matrix=cv2.getRotationMatrix2D(center,angle,scale=1.0)
            rotated_image=cv2.warpAffine(self.img,rotation_matrix,(width,height))
            cv2.imwrite("rotated_augmented_image.jpg",rotated_image)
       
        #Saturation augmentation
            hsv=cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV).astype(np.float32)
            saturation_factor=1.5 
            hsv[:,:,1]=np.clip(hsv[:,:,1]*saturation_factor,0,255).astype(np.uint8)
            saturated_img=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            cv2.imwrite("saturation_augmented_image.jpg",saturated_img)
        #gaussian_noise augmentation:
            mean=0
            std=15
            gaussian_noise=np.random.normal(mean,std,self.img.shape)
            noised_img=self.img.astype(np.float32)
            noised_img=np.clip(noised_img+gaussian_noise,0,255).astype(np.uint8)
            cv2.imwrite("gaussian_noise_augmented_image.jpg",noised_img) 
            
        #translated_image augmentation
            tx=50
            ty=150
            translated_matrix=np.array([[1,0,tx],[0,1,ty]]) #if instead  of 0 if shear_factor put it becomes sheared image
            translated_img=cv2.warpAffine(self.img.astype(np.uint8),translated_matrix.astype(np.float32),(width,height))
            cv2.imwrite("translated_image.png",translated_img)
            
        #zoomed transformation:
            fx=1.5
            fy=1.5
            scaled=cv2.resize(self.img,None,fx=fx,fy=fy).astype(np.float32)
            (start_y)=(scaled.shape[0]-self.img.shape[0])//2
            start_x=(scaled.shape[1]-self.img.shape[1])//2
            self.img=self.img.astype(np.uint8)
            zoomed_img=scaled[(start_y):(start_y)+self.img.shape[0],(start_x):start_x+self.img.shape[1]].astype(np.uint8)
            cv2.imwrite("zoomed_image.jpg",zoomed_img)
        #flipping:
            self.img=self.img.astype(np.uint8)
            hflip=cv2.flip(self.img,1)
            cv2.imwrite("horizontal_flipped_image.jpg",hflip)
            vflip=cv2.flip(self.img,0)   
            cv2.imwrite("vertical_flipped_image.jpg",vflip)
            bflip=cv2.flip(self.img,-1)
            cv2.imwrite("both_flipped_image.jpg",bflip)
            hflip=cv2.cvtColor(hflip,cv2.COLOR_BGR2RGB)
            vflip=cv2.cvtColor(vflip,cv2.COLOR_BGR2RGB)
            bright_img=cv2.cvtColor(bright_img,cv2.COLOR_BGR2RGB)
            contrast_img=cv2.cvtColor(contrast_img,cv2.COLOR_BGR2RGB)
            rotated_image=cv2.cvtColor(rotated_image,cv2.COLOR_BGR2RGB)
            zoomed_img=cv2.cvtColor(zoomed_img,cv2.COLOR_BGR2RGB)
            gaussian_noise=cv2.cvtColor(gaussian_noise.astype(np.uint8),cv2.COLOR_BGR2RGB)
            saturated_img=cv2.cvtColor(saturated_img,cv2.COLOR_BGR2RGB)
            
            
            plt.figure(figsize=(8,4))
            plt.subplot(2,4,1)
            plt.imshow(bright_img)
            plt.title("bright image")
            plt.axis("off")
           
            plt.subplot(2,4,2)
            plt.imshow(contrast_img)
            plt.title("contrast image")
            plt.axis("off")
         
            plt.subplot(2,4,3)
            plt.imshow(rotated_image)
            plt.title("rotated image")
            plt.axis("off")
          
            plt.subplot(2,4,4)
            plt.imshow(hflip)
            plt.title("horizontal flipped image")
            plt.axis("off")
           
            plt.subplot(2,4,5)
            plt.imshow(vflip)
            plt.title("vertical flipped image")
            plt.axis("off")
       
            plt.subplot(2,4,6)
            plt.imshow(zoomed_img)
            plt.title("zoomed image")
            plt.axis("off")
            
        
            
            plt.subplot(2,4,7)
            plt.imshow(cv2.cvtColor(translated_img,cv2.COLOR_BGR2RGB))
            plt.title("translated image")
            plt.axis("off")
            
            plt.subplot(2,4,8)
            plt.imshow(saturated_img)
            plt.title("saturated image")
            plt.axis("off")
            plt.show()
        
        
def main():
    image_path=input("Enter path:")
    ac=augmentation(image_path)
    ac.augmentation()
main()  
import cv2
import os
def rotated_image(image,angle):
    if os.path.exists(image) == False:
            raise FileNotFoundError("Image file not found.")
    if os.path.splitext(image)[1].lower() not in ('.png','.jpg','.jpeg','.bmp'):
            raise ValueError("Unsupported image format.")
    img=cv2.imread(image)
    (h,w)=img.shape[:2]
    center=(w//2,h//2)
    M=cv2.getRotationMatrix2D(center,angle,1.0)
    rotated=cv2.warpAffine(img,M,(w,h))
    return rotated
class keypoint_detection():
    def __init__(self,image_path1,image2):
        
        if os.path.exists(image_path1) == False:
            raise FileNotFoundError("Image file not found.")
        if os.path.splitext(image_path1)[1].lower() not in ('.png','.jpg','.jpeg','.bmp'):
            raise ValueError("Unsupported image format.")
        self.img2=image2
        self.img1=cv2.imread(image_path1,cv2.IMREAD_GRAYSCALE)
    def detection(self):
        orb=cv2.ORB_create(nfeatures=480,nlevels=9,scaleFactor=1.3,edgeThreshold=31)
        kp1,desc1=orb.detectAndCompute(self.img1,None)
        kp2,desc2=orb.detectAndCompute(self.img2,None)
        bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches=bf.match(desc1,desc2)
        matches=sorted(matches,key=lambda x:x.distance)
        good_matches=[]
        for m in matches:
            if m.distance<50:
                good_matches.append(m)
        img_matched=cv2.drawMatches(self.img1,kp1,self.img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("ORB Keypoint Matches",img_matched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
def main():
    image1=input("Enter the path for the first image: ")
    image2=rotated_image(image1,45)
    detector=keypoint_detection(image1,image2)  
    detector.detection()
main()
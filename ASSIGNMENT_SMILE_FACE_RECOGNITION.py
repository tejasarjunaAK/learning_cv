import cv2
import numpy as np


class Recognition():
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
        self.face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')      #loads cascade once that is enough 
        self.smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
    def face_recognition(self,gray,frame_copy):  
        faces=self.face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=4)     
        for face in faces:
            (x,y,w,h)=face
            cv2.rectangle(frame_copy,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame_copy,"Face",(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            
            eyes=self.eye_cascade.detectMultiScale(gray[y:y+h,x:x+w],scaleFactor=1.3,minNeighbors=5)
            for (ex,ey,ew,eh) in eyes:# for detecting multiple eyes inside face rectangle
              cv2.rectangle(frame_copy,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
              cv2.putText(frame_copy,"Eye",(x+ex,y+ey-10),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            for (sx,sy,sw,sh) in self.smile_cascade.detectMultiScale(gray[y:y+h,x:x+w],scaleFactor=1.8,minNeighbors=10):
              cv2.rectangle(frame_copy,(x+sx,y+sy),(x+sx+sw,y+sy+sh),(0,0,255),2)
              cv2.putText(frame_copy,"Smile",(x+sx,y+sy-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
        return frame_copy
            
        
    def frame_capture(self):
        while True:
            ret,frame=self.cap.read()
            if not ret:
                break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame_copy=frame.copy()
            final= self.face_recognition(gray,frame_copy)
            cv2.imshow("FACE RECOGNITION",final)
            if cv2.waitKey(100)& 0xFF==ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
            
def main():
    print("webcam")
    fc=Recognition()
    fc.frame_capture()
    
main()
    

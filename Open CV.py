#Open CV
import cv2 
import numpy as np
''' whenever image is imread() then by default it reads in BGR format
 even if the original image is in grayscale style.
 -> we can read image and write it to a new file with different type
 from jpg to png by writing to file with corresponding extension
 
 -> this is the format'''
image= cv2.imread('me.jpg')
#imread ('filename')checks only for jpg or png files inside that folder
'''if we want it to use files outside folder also 
then we have to write full path name'''
# for getting path Ctrl+Shift+C or copy as path option when right click
image =cv2.imread(r"C:\Users\Tejas Arjuna\OneDrive\Pictures\me .jpg")
'''-> for writing into image whether new file or overwriting over already present'''
cv2.imwrite('me.png',image)
''' -> for reading image as grayscale or bgr 
-> we use cv2.CV_LOAD_IMAGE_COLOR(BGR) or cv2.CV_LOAD_IMAGE_GRAYSCALE(grayscale) or 
CV_LOAD_IMAGE_UNCHANGED'''
grayer=cv2.imread('me.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
'''-> for getting bgr array for particular pixel in the image that we have read
variable_name=image[100,50] or image[100,50,3] for getting bgr values for that pixelin 1 numpy array
->for getting only blue or green or red value ( blue indexx of array :0,g:1,r:2) 
blue=image[100,50,0]
green=image[100,50,1]
red=image[100,50,2]
->Even if image is bgr if we want the grayscale value of image we do
gray_value = 0.114 * b + 0.587 * g + 0.299 * r'''
# Whenever image is read in cv2 it stores the image as numpy byte array: if bgr image array is 3d
#if grayscale image array is 2d array
'''-> BGR: each row is an array; each pixel has 3 values in array(each with 0-255)
-> grayscale: each row is array; each pixel is only single value(from 0-255)'''
#to convert from bgr to grayscale and vice versa
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
new_image=cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
#Toget shape
image.shape
gray_image.shape
#to change shape
image.reshape(640,300,3)
gray_image.reshape(540,280)
#to flatten array to 1d array
image.reshape(-1,3) #for bgr image -> this will give reshaped bgr image
gray_image.reshape(-1)
# easily numpy array can be converted to image
cv2.imwrite('me.png',numpy_array_name)
# to create flat random integer array in numpy from 0-255. We used uint8 and not int only for saving memory
arr2 = np.random.randint(256, size=120000, dtype=np.uint8)
cv2.imwrite('me.png',arr2.reshape(400,100,3))
cv2.imwrite('me.png',arr2.reshape(400,300))# This is 300x400 size image (length 300; height 400)
#To display image 
cv2.imshow('My Image Window', image)
cv2.waitKey(0) # if cv2.waitkey(2) then window gets destroyed automatically after 2 seconds
cv2.destroyAllWindows()

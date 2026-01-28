import cv2
import numpy as np
import pandas as pd

#min-max normalization:
image=cv2.imread(r"C:\Users\Tejas Arjuna\OneDrive\Pictures\tejas_anime_portrait.png")
#cv2.imshow("image",image)
cv2.waitKey(0)
normalized=image.astype(np.float32)/255.0
print(normalized)
max_=np.max(normalized)
min=np.min(normalized)
normalized=(normalized.astype(np.float32)-min)/(max_-min)
print("after min max normalization")
print(normalized)


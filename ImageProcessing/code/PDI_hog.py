# player detection with HOG

import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

original = cv2.imread("res/img_01.jpg")
img = original.copy()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

regions, weights = hog.detectMultiScale(img, winStride=(4,4), padding=(8, 8), scale=1.05)
print(regions)

for (x,y,w,h) in regions:
    cv2.rectangle(original, (x,y), (x+w,y+h), (0,0,255), 1)

cv2.imshow("img", original)
cv2.waitKey(0)
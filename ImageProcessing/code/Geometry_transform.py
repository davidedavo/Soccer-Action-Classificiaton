import cv2
import numpy as np

img = cv2.imread("res/img_01.jpg")
field = cv2.imread("res/field2.jpg")

cv2.circle(img, (651,164), 2, (0,0,255),2)
cv2.putText(img, "1", (651+10,164), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (468,248), 2, (0,0,255),2)
cv2.putText(img, "2", (468+10,248), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (110,205), 2, (0,0,255),2)
cv2.putText(img, "3", (110+10,205), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
cv2.circle(img, (145,126), 2, (0,0,255),2)
cv2.putText(img, "4", (145+10,126), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

cv2.circle(field, (118,115), 2, (255,0,0),2)
cv2.putText(field, "1", (118+10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (119, 176), 2, (255,0,0),2)
cv2.putText(field, "2", (119+10, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (67, 156), 2, (255,0,0),2)
cv2.putText(field, "3", (67+10, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
cv2.circle(field, (37,115), 2, (255,0,0),2)
cv2.putText(field, "4", (37+10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

pts_src = np.array([[651, 164], [468, 248], [110, 205], [145, 126]])
pts_dst = np.array([[118, 115], [119, 176], [67, 156], [37, 115]])

H, status = cv2.findHomography(pts_src, pts_dst)

point = np.array([[480, 320]], np.float32)
point = np.array([point])

point_in_field = cv2.perspectiveTransform(point, H)
outX = round(point_in_field[0][0][0])
outY = round(point_in_field[0][0][1])

cv2.circle(img, (480, 320), 2, (255,0,255),2)
cv2.putText(img, "P", (480+10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)
cv2.circle(field, (outX, outY), 2, (255,0,255),2)
cv2.putText(field, "P", (outX+10,outY), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)

cv2.imshow("Img", img)
cv2.imshow("Field", field)
cv2.waitKey(0)
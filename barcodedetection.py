import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help = "Path to The Image File")
args= vars(ap.parse_args())
#loading the Image
image = cv2.imread(args["image"])
# GrayScale Conversion of Image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Compute the Scharr Gradient Magnitude
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray,ddepth=ddepth,dx=1,dy=0,ksize=-1)
gradY = cv2.Sobel(gray,ddepth=ddepth,dx=0,dy=1,ksize=-1)

# Subract Y-Gradient From X-Gradient
gradient = cv2.subtract(gradX,gradY)
gradient = cv2.convertScaleAbs(gradient)

# Apply Blur & Thresholding the Image
blurred = cv2.blur(gradient, (9,9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
# Apply Kernel to The Image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
# performing dilation & erosion
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

#finding the contours in the Image & then Sorting
cnts = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# Compute the Rotating bounding Box of the Largest Contour
rect = cv2.minAreaRect(c)
box = cv2.cv.boxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(image, [box], -1, (0 , 0 , 255), 3)
cv2.imshow("Image",image)
cv2.waitKey(0)

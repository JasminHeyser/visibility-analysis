# import the necessary packages
import argparse
import cv2
import numpy as np
import pywt
import scipy
from matplotlib import pyplot as plt
from helpers.rescale import rescaleimage
import FocusMethods
import TexturMethods


img = cv2.imread('C:\\Users\\jheys\Documents\\01_BA\VSCode\\20220316-07-46-50-759188_a_matrix_7_SEP_delta102_x100_theta0_do1_du0.65__orig.jpg', 0) # load an image

## rescales images
rescimg= rescaleimage(img)
cv2.imshow('masked image', rescimg)

##create maske and mask the image
mask = np.zeros(rescimg.shape[:2], dtype="uint8")
cv2.rectangle(mask, (200, 820), (100, 150), 255, -1)

cv2.imshow("Rectangular Mask", mask)


# Define an array of endpoints of polygon
# later with json !!


#points = np.array([[220, 120], [130, 200], [130, 300],
 #                  [220, 380], [310, 300], [310, 200]])

#cv.fillPoly(mask, pts=[points], color=(0, 255, 0))

maskedimg = cv2.bitwise_and(rescimg,rescimg,mask = mask)

cv2.imshow('masked image', maskedimg)


cv2.waitKey(0)
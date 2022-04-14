# import the necessary packages
import argparse
import cv2
from cv2 import mean
import numpy as np
import pywt
import scipy
from matplotlib import pyplot as plt
#from rescale import rescaleimage
#from DFT import calcdft

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-s", "--scharr", type=int, default=0,
	help="Schaar or Sobel Operator")
args = vars(ap.parse_args())
"""

def calcschaarmMAG(image):

# set the kernel size, depending on whether we are using the Sobel
# operator of the Scharr operator, then compute the gradients along
# the x and y axis, respectively
	ksize = -1 #if args["scharr"] > 0 else 3
	gX = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=1, dy=0)
	gY = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=0, dy=1)


# the gradient magnitude images are now of the floating point data
# type, so we need to take care to convert them back a to unsigned
# 8-bit integer representation so other OpenCV functions can operate
# on them and visualize them
	gX = cv2.convertScaleAbs(gX)
	gY = cv2.convertScaleAbs(gY)



# combine the gradient representations into a single image
	combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

# calculates the magnitude (intesity of the edge)
	magnitude = np.sqrt((gX ** 2) + (gY ** 2))
	mean_magnitude = np.mean(magnitude)

	return mean_magnitude

# prints the mean of the combined Schaar filter values and shows output images
	"""
	print('mean magnitude of combinde schaar:',mean_magnitude)

# show our output images
cv2.imshow("Scharr X", gX)
cv2.imshow("Scharr Y", gY)
cv2.imshow("Scharr Combined", combined)
cv2.waitKey(0)
"""
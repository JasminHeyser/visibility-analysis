# import the necessary packages
import argparse
import cv2
import numpy as np
import pywt
import scipy
from matplotlib import pyplot as plt




def calcsobelMAG(image):
# compute gradients along the x and y axis, respectively
	gX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
	gY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
# compute the gradient magnitude and orientation
	magnitude = np.sqrt((gX ** 2) + (gY ** 2))
	orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
	mean_magnitude = np.mean(magnitude)
	return mean_magnitude

# initialize a figure to display the input grayscale image along with
# the gradient magnitude and orientation representations, respectively
"""
	(fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
# plot each of the images
	axs[0].imshow(image, cmap="gray")
	axs[1].imshow(magnitude, cmap="jet")
	axs[2].imshow(orientation, cmap="jet")
# set the titles of each axes
	axs[0].set_title("Grayscale")
	axs[1].set_title("Gradient Magnitude")
	axs[2].set_title("Gradient Orientation [0, 180]")
# loop over each of the axes and turn off the x and y ticks
	for i in range(0, 3):
		axs[i].get_xaxis().set_ticks([])
		axs[i].get_yaxis().set_ticks([])
# show the plots
	plt.tight_layout()
	plt.show()
"""


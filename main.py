# import the necessary packages
import argparse
import cv2
import numpy as np
import pywt
import sys
import scipy
from matplotlib import pyplot as plt
from FocusMethods.DCT import calcdct
from FocusMethods.DFT import calcdft
from FocusMethods.TEN import calctenengrad
from FocusMethods.LAP import diagonal_laplacian,modified_laplacian,variance_of_laplacian
from FocusMethods.STAT import histogram_range,grayvalue_variance
from FocusMethods.WAV import sum_of_wavelet_coefficients, ratio_of_wavelet_coefficients, variance_of_wavelet_coefficients
from Magnitude_Schaar import calcschaarmMAG
from gabor import gabor
from Magnitude_Sobel import calcsobelMAG
from helpers.rescale import rescaleimage
from helpers.maskimages import maskimage
#from DFT import calcdft

#   load an image in grayscale!
img = cv2.imread('C:/Users/jheys/Documents/01_BA/Bilddaten/AnnoBilder_IDC/20220317-16-20-28-116527_a_matrix_79_IDC_delta218_x13_theta0_do1_du0.3__orig.jpg', cv2.IMREAD_GRAYSCALE)
img = rescaleimage(img,scale=0.3)



#   mask the image
#   load mask image
mask = cv2.imread('C:/Users/jheys/Documents/01_BA/exported_Toolima_masks/20220317-16-20-28-116527_a_matrix_79_IDC_delta218_x13_theta0_do1_du0.3__orig___fullmask.pgm',cv2.IMREAD_UNCHANGED)
mask = rescaleimage(mask,scale=0.3)
ret,binmask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
cv2.imshow('Mask Image', binmask)


maskedimg= maskimage(img,binmask)
#   show masked image
cv2.imshow('Masked Image', maskedimg)

'''
#   calculate the DCT
dct = calcdct(img)
#print('DCT Array/n' , dct)

#   calculate the DFT
dft = calcdft(img)
#print('DFT Array/n' , dct)


#   calculate tenengrad and tenengrad variance
tenengrad    = calctenengrad(img)
tenengradvar = tenengrad.var()
#print("tenengrad/n" , tenengrad)
#print("tenengrad variance/n" , tenengradvar)

#   Lapcian values

varlap = variance_of_laplacian(img)
#print("variance_of_laplacian:",varlap)

#reason why variance of laplace from leos thesis
"""
the reasoning for this measure is that if there is a lot of variance in the approximation
of the second derivative of the image (i.e. the Laplacian), then there should
be many edges in the original image. It is implied, that if there are a lot of edges in
the image, that the image is sharp.

"""

modlap=modified_laplacian(img)
#print("modified_laplacian:",modlap)


dialap=diagonal_laplacian(img)
#print("diagonal_laplacian:",dialap)


histrange =histogram_range(img)
#print("histogramm range",histrange)


grayvalue = grayvalue_variance(img)
#print("grayvalue_variance",grayvalue)


#calculating wavelets
sum_of_wavelet_coefficients = sum_of_wavelet_coefficients(img)
#print("sum_of_wavelet_coefficients",sum_of_wavelet_coefficients)

variance_of_wavelet_coefficients = variance_of_wavelet_coefficients(img)
#print("ratio_of_wavelet_coefficients",ratio_of_wavelet_coefficients)

ratio_of_wavelet_coefficients = ratio_of_wavelet_coefficients(img)
#print("ratio_of_wavelet_coefficients",ratio_of_wavelet_coefficients)

Schaarmagnitude = calcschaarmMAG(img)
#print("Mean Magnitude of Shaar operator",Schaarmagnitude)

Sobelmagnitude = calcsobelMAG(img)
#print("Mean Magnitude of Sobel operator",Sobelmagnitude)
'''

#   applying Gabor Filter
'''vermutlich nicht anwendbar weil sehr richtungs abhängig. Geeignet um eine  Gabort filter liste zu erstellen und diese in ML zu nutzen'''
#gaborimg= gabor(img)
#cv2.imshow("gabor Image" , gaborimg)


#   Save image
#cv2.imwrite('C:/Users/jheys/Documents/01_BA/VSCode/maskedwithhorse.png', dst)

cv2.waitKey(0)







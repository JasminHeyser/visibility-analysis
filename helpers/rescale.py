
# import the necessary packages
import argparse
import cv2
import numpy as np
import pywt
import scipy
from matplotlib import pyplot as plt
#from rescale import rescaleimage
#from DFT import calcdft



# resize image

def rescaleimage(img,scale):
    width = int(img.shape[1]* scale)
    hight = int(img.shape[0]* scale)
    dimensions = (width, hight)
    return cv2.resize(img,dimensions, interpolation=cv2.INTER_AREA)

    
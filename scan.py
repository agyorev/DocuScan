#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import cv2
import numpy as np
import argparse

from transform import Transform
from basic_image import BasicImage
from combine_images import CombineImages

""" Arugment Parser """
ap = argparse.ArgumentParser()
ap.add_argument('-i',
    '--image',
    required = True,
    help     = 'path to the image')

ap.add_argument('-H',
    '--height',
    required = False,
    default  = 300,
    help     = 'height of the image image we will process and use for finding the contours (default: 300)')

ap.add_argument('-n',
    '--noise',
    required = False,
    default  = 0,
    help     = 'the level to which we remove noise and smaller details from the scan (default: 0, i.e. preserve everything')

ap.add_argument('-c',
    '--closing',
    required = False,
    default  = 3,
    help     = 'the size of the closing element after applying the Canny edge detector')
args         = vars(ap.parse_args())

# Getting the user input
HEIGHT              = int(args['height'])
NOISE_REMOVAL_LEVEL = max(int(args['noise']) * 2 - 1, 0)
CLOSING_SIZE        = int(args['closing'])
bi                  = BasicImage(args['image'])

original   = bi.get().copy()
ratio      = original.shape[0] / float(HEIGHT)
image      = bi.resize('H', HEIGHT)
total_area = image.shape[0] * image.shape[1]

#BasicImage(image).show()

""" Step 1: Edge Detection """
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # get the grayscale image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#gray = cv2.GaussianBlur(gray, (3, 3), 0) # with a bit of blurring
#BasicImage(gray).show()

# automatic Canny edge detection thredhold computation
high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_thresh = high_thresh / 2.0

edged = cv2.Canny(gray, low_thresh, high_thresh) # detect edges (outlines) of the objects
#BasicImage(edged).show()

# since some of the outlines are not exactly clear, we construct
# and apply a closing kernel to close the gaps b/w white pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_SIZE, CLOSING_SIZE))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#BasicImage(closed).show()

""" Step 2: Finding Contours """
(contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# looping over the contours found
approx_all = []
for contour in contours:
    # approximating the contour
    contour = cv2.convexHull(contour)
    peri    = cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, 0.02 * peri, True)
    area    = cv2.contourArea(contour)

    # we don't consider anything less than 5% of the whole image
    if area < 0.05 * total_area:
        continue

    # if the approximated contour has 4 points, then assumer it is a book
    # a book is a rectangle and thus it has 4 vertices
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        approx_all.append(approx)
        total += 1

print 'Found %d books/papers in the image.' % total
#BasicImage(image).show()

""" Step 3: Apply a Perspective Transform and Threshold """
for approx in approx_all:
    warped = Transform.get_box_transform(original, approx.reshape(4, 2) * ratio)
    #BasicImage(warped).show()

    scan_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scan_warped = cv2.medianBlur(scan_warped, NOISE_REMOVAL_LEVEL)
    scan_warped = cv2.adaptiveThreshold(scan_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #BasicImage(scan_warped).show()

""" Displaying all intermediate steps into one image """
top_row = CombineImages(300, original, gray)
bot_row = CombineImages(300, closed, image)
com_img = np.vstack((top_row, bot_row))
BasicImage(com_img).show()

BasicImage(CombineImages(700, warped, scan_warped)).show()

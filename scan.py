#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import cv2
import numpy as np
import argparse

from transform import Transform
from basic_image import BasicImage

""" Arugment Parser """
ap = argparse.ArgumentParser()
ap.add_argument('-i',
    '--image',
    required = True,
    help     = 'path to the image')

ap.add_argument('-H',
    '--height',
    required = False,
    default  = 500,
    help     = 'height of the image image we will process and use for finding the contours (default: 500)')

ap.add_argument('-n',
    '--noise',
    required = False,
    default  = 0,
    help     = 'the level to which we remove noise and smaller details from the scan (default: 0, i.e. preserve everything')
args         = vars(ap.parse_args())

# Getting the user input
HEIGHT              = int(args['height'])
NOISE_REMOVAL_LEVEL = max(int(args['noise']) * 2 - 1, 0)
bi                  = BasicImage(args['image'])

original = bi.get().copy()
ratio    = original.shape[0] / float(HEIGHT)
image    = bi.resize('H', HEIGHT)

""" Step 1: Edge Detection """
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # get the grayscale image
gray = cv2.GaussianBlur(gray, (3, 3), 0) # with a bit of blurring
#BasicImage(gray).show()

edged = cv2.Canny(gray, 10, 250) # detect edges (outlines) of the objects
#BasicImage(edged).show()

# since some of the outlines are not exactly clear, we construct
# and apply a closing kernel to close the gaps b/w white pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#BasicImage(closed).show()

""" Step 2: Finding Contours """
(contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# looping over the contours found
approx_all = []
for contour in contours:
    # approximating the contour
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # if the approximated contour has 4 points, then assumer it is a book
    # a book is a rectangle and thus it has 4 vertices
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        approx_all.append(approx)
        total += 1

print 'Found %d books/papers in the image.' % total
BasicImage(image).show()

""" Step 3: Apply a Perspective Transform and Threshold """
for approx in approx_all:
    warped = Transform.get_box_transform(original, approx.reshape(4, 2) * ratio)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.medianBlur(warped, NOISE_REMOVAL_LEVEL)
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    BasicImage(warped).show()

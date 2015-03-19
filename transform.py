#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import cv2
import numpy as np

class Transform(object):
    def __init__(self):
        pass

    def get_points_order(self, _points):
        box        = np.zeros((4, 2), dtype='float64') # order will be: TL, TR, BR, BL

        coord_sum  = _points.sum(axis = 1)
        box[0]     = _points[np.argmin(coord_sum)]     # TL - has the min sum
        box[2]     = _points[np.argmax(coord_sum)]     # BR - has the max sum

        coord_diff = np.diff(_points, axis = 1)
        box[1]     = _points[np.argmin(coord_diff)]    # TR - has the min diff
        box[3]     = _points[np.argmax(coord_diff)]    # BL - has the max diff

        return box                                     # return the ordered coordinates

    def get_box_transform(self, _image, _points):
        init_box         = get_points_order(_points)
        (tl, tr, br, bl) = init_box # get the correct order

        width_top    = np.sqrt(((tl[0] - tr[0]) ** 2) + ((tl[1] - tr[1]) ** 2)) # distance b/w TL and TR
        width_bot    = np.sqrt(((bl[0] - br[0]) ** 2) + ((bl[1] - br[1]) ** 2)) # distance b/w BL and BR
        max_width    = max(int(width_top), int(width_bot))                      # the width of the new image

        height_left  = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # distance b/w TL and BL
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) # distance b/w TR and BR
        max_height   = max(int(height_left), int(height_right))                 # the height of the new image

        dest_box = np.array([ # the resulting edge points after the transform
            [            0, 0              ], # TL
            [max_width - 1, 0              ], # TR
            [max_width - 1, max_height - 1 ], # BR
            [            0, max_height - 1]], # BL
            dtype='float64')

        M     = cv2.getPerspectiveTransform(init_box, dest_box)         # transformation matrix
        image = cv2.warpPerspective(_image, M, (max_width, max_height)) # apply the transform

        return image # return the warped image

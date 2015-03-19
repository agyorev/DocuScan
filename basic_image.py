#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import cv2
import numpy as np

class BasicImage(object):
    def __init__(self, _image_path):
        self.image = cv2.imread(_image_path)

    def get_size(self):
        return self.image.shape

    def show(self):
        cv2.imshow('Basic Image', self.image)
        cv2.waitKey(0)

    def save(self, _image_path):
        cv2.imwrite(_image_path, self.image)

    def resize(self, _type, _size):
        if _type == 'w' or _type == 'W':
            ratio = float(_size) / self.image.shape[1]
            dim   = (_size, int(self.image.shape[0] * ratio))
        elif _type == 'h' or _type == 'H':
            ratio = float(_size) / self.image.shape[0]
            dim   = (int(self.image.shape[1] * ratio), _size)
        else:
            return self.image

        self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

    def rotate(self, _angle):
        (height, width) = self.image.shape[:2]
        center          = (width / 2, height / 2)

        # make it a 3x3 rotation matrix
        M = np.vstack([cv2.getRotationMatrix2D(center, _angle, 1.0), [0, 0, 1]])
        R = np.matrix(M[0:2, 0:2])

        half_height = height / 2.0
        half_width  = width / 2.0

        # coordinates of the rotated corners
        rotated_corners = [
                (np.array([-half_width, half_height]) * R).A[0],
                (np.array([ half_width, half_height]) * R).A[0],
                (np.array([-half_width,-half_height]) * R).A[0],
                (np.array([ half_width,-half_height]) * R).A[0]]

        # new image size
        x_coords = [point[0] for point in rotated_corners]
        y_coords = [point[1] for point in rotated_corners]

        right_bound = max(x_coords)
        left_bound  = min(x_coords)
        top_bound   = max(y_coords)
        bot_bound   = min(y_coords)

        new_height = int(abs(top_bound - bot_bound))
        new_width  = int(abs(right_bound - left_bound))

        # translation matrix to keep it centered
        T = np.matrix([
            [1, 0, int(new_width / 2.0 - half_width)],
            [0, 1, int(new_height / 2.0 - half_height)],
            [0, 0, 1]])

        # combining rotation and translation
        M = (np.matrix(T) * np.matrix(M))[0:2, :]

        self.image = cv2.warpAffine(self.image, M, (new_width, new_height), flags = cv2.INTER_LINEAR)

    def crop(self, _top, _bot, _left, _right):
        self.image = self.image[_top:_bot + 1, _left:_right + 1]


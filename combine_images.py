import cv2
import numpy as np

from basic_image import BasicImage

def CombineImages(_height, *args):
    result = ()

    for img in args:
        img = BasicImage(img).resize('H', _height)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        result = result + (img,)

    result = np.hstack(result)

    return result

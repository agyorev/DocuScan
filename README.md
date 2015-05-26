# DocuScan
An OpenCV project for detecting books, papers, or any rectangular objects in an image, regardless the perspective, and straightening it as if it was a normal scan.

## Requirements
[Python 2.7.9](https://www.python.org/downloads/release/python-279/)

[OpenCV](http://opencv.org)

[NumPy](http://www.numpy.org)

## Results
Images from example runs can be found in the [results](./results) folder.

#### Exampes

##### Automaticly (heuristically) setting the argument parameters (height and closing) to find the best fit. 
We have set a noise removal level of 3.


    $> python2.7 scan.py -i img/example9.jpg -a -n 3


##### Manual argument parameter setting


    $> python2.7 scan.py -i img/example10.jpg -H 400 -c 1 -n 2


- Detection steps
    1. Original image
    2. Bilateral filter
    3. Canny edges + Morphological closing
    4. Contour detection

![example10](https://raw.githubusercontent.com/agyorev/DocuScan/master/results/example10_400_1.jpg)

- Fixing the perspective and applying adaptive thresholding

![example10a](https://raw.githubusercontent.com/agyorev/DocuScan/master/results/example10_400_1_2_a_final.jpg)
![example10b](https://raw.githubusercontent.com/agyorev/DocuScan/master/results/example10_400_1_2_b_final.jpg)

    $> python2.7 scan.py -i img/example5.jpg -H 600 -c 3

- Detection steps

![example5](https://raw.githubusercontent.com/agyorev/DocuScan/master/results/example5_600_3.jpg)

- Fixing the perspective and applying adaptive thresholding

![example5](https://raw.githubusercontent.com/agyorev/DocuScan/master/results/example5_600_3_final.jpg)

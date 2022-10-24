# Code: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
# Explanation: https://theailearner.com/tag/cv2-warpaffine/

import os
import cv2
import numpy as np

# read the original Messi image
image = cv2.imread("Messi.jpg")
print(image.shape)

# image resize using cv2.resize function
# note: cvSize(width, height)
height, width = image.shape[:2]
resized = cv2.resize(image, (int(0.5*width), int(0.5*height)))
print(resized.shape)

# affine transform matrix
M = np.array(
    [[0.5, 0, 100],
     [0, 0.5, 50]], dtype=np.float32
)
# print(M)
image2 = cv2.warpAffine(image, M, (width, height))
cv2.imshow("translated", image2)
cv2.waitKey(0)
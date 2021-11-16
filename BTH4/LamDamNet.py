# -------------------------------- Lam dam net ky tu --------------------------------------------

import cv2
from IPython.display import Image
image = cv2.imread('test2.png', cv2.IMREAD_GRAYSCALE)  #doc hinh anh
img = cv2.convertScaleAbs(image, 1.1, 5)
cv2.imwrite('test0000.png', img)
Image('test00001111')



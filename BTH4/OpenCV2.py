import numpy as np
import cv2
from matplotlib import pyplot as plt
# Image operation using thresholding
img = cv2.imread('C:\\Users\\Dell 3567\\Desktop\\Xulyanh\\BTH4\\test4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Org image', thresh)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
# kernel = np.ones((5, 5), np.uint8)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
# kernel, iterations = 2)
# ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)
# cv2.imshow('image', fg)
# k = cv2.waitKey(0)
# if k == 27: # wait for ESC key to exit
#     cv2.destroyAllWindows()
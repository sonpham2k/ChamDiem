import cv2
from IPython.display import Image
image = cv2.imread('test2.png', cv2.IMREAD_GRAYSCALE)  #doc hinh anh
#có thể áp dụng các bộ lọc  sepFilter2D(), filter2D(), blur(), boxFilter(), bilateralFilter(), medianBlur()
#áp dụng bộ lọc trung vi 3x3

img = cv2.medianBlur(image, 9) 
cv2.imwrite('test.png', img) #lưu ảnh vào đường dẫn
Image('Hinh Anh')  #hiển thị ảnh
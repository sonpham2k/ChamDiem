# import các thư viện cần thiết
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# thiết lập tham số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
 help="path to the input image")
args = vars(ap.parse_args())

# thiết lập từ khóa cho câu trả lời
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# load ảnh, chuyển sang định dạng gray và dùng phép mờ ảnh blur
# làm mỏng và tìm cạnh
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# tìm contours trong edge map, sau đó khởi tạo
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
 cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# phải chắc rằng có nhiều hơn 1 contour được tìm thấy
if len(cnts) > 0:
 # săp xếp các contour tìm được
 # theo thứ tự lớn tới bé
 cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

 # loop over the sorted contours
 for c in cnts:
  # approximate contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)

  # nếu approximated contour lớn hơn 4 điểm
  # thì nó chính là 4 góc của bài trắc nghiệm
  if len(approx) == 4:
   docCnt = approx
   break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# sử dụng phương pháp Otsu's thresholding 
# piece of paper
thresh = cv2.threshold(warped, 0, 255,
 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

 # find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
 cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# loop over the contours
for c in cnts:
 # compute the bounding box of the contour, then use the
 # bounding box to derive the aspect ratio
 (x, y, w, h) = cv2.boundingRect(c)
 ar = w / float(h)

 # in order to label the contour as a question, region
 # should be sufficiently wide, sufficiently tall, and
 # have an aspect ratio approximately equal to 1
 if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
    questionCnts.append(c)

  # sắp xếp các contours câu hỏi từ trên xuống dưới sau đó khởi tạo


# tổng số các câu trả lời đúng
questionCnts = contours.sort_contours(questionCnts,
method="top-to-bottom")[0]
correct = 0

# mỗi câu hỏi có thể có 5 câu trả lời
# cần loop 5 lần
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
 # xắp xếp các câu trả lời từ trái sang phải
 cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
 bubbled = None

# loop over the sorted contours
 for (j, c) in enumerate(cnts):
  # construct a mask that reveals only the current
  # "bubble" for the question
  mask = np.zeros(thresh.shape, dtype="uint8")
  cv2.drawContours(mask, [c], -1, 255, -1)

  # apply the mask to the thresholded image, then
  # count the number of non-zero pixels in the
  # bubble area
  mask = cv2.bitwise_and(thresh, thresh, mask=mask)
  total = cv2.countNonZero(mask)

  # if the current total has a larger number of total
  # non-zero pixels, then we are examining the currently
  # bubbled-in answer
  if bubbled is None or total > bubbled[0]:
   bubbled = (total, j)

# initialize the contour color and the index of the
 # *correct* answer
 color = (0, 0, 255)
 k = ANSWER_KEY[q]

 # check to see if the bubbled answer is correct
 if k == bubbled[1]:
  color = (0, 255, 0)
  correct += 1

 # draw the outline of the correct answer on the test
 cv2.drawContours(paper, [cnts[k]], -1, color, 3)

 # grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
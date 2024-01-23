import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

line = cv2.imread('line.png', cv2.IMREAD_GRAYSCALE)
h, w = line.shape

half = w // 2

left_line = line[:, :half]
right_line = line[:, half:]

line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)

left_lane_x = []
left_lane_y = []

right_lane_x = []
right_lane_y = []

mid_lane_x = []
mid_lane_y = []

for i in range(h - 1, -1, -10):
    left_midpts = np.nonzero(left_line[i])[0]
    right_midpts = np.nonzero(right_line[i])[0] + half

    if len(left_midpts):
        left_midpt = np.sum(left_midpts) // len(left_midpts)
        #print('left : ', left, end=' ')
        cv2.circle(line, (left_midpt, i), 3, (0, 0, 255), -1)
        left_lane_x.append(left_midpt) # x, y
        left_lane_y.append(i) # x, y

    if len(right_midpts):
        right_midpt = np.sum(right_midpts) // len(right_midpts)
        #print('right : ', right, end=' ')
        cv2.circle(line, (right_midpt, i), 3, (0, 255, 0), -1)
        right_lane_x.append(right_midpt) # x, y
        right_lane_y.append(i) # x, y

    if len(left_midpts) and len(right_midpts):
        mid_lane_x.append(left_midpt + right_midpt >> 1)
        mid_lane_y.append(i)

left_lane_x = np.array(left_lane_x)
left_lane_y = np.array(left_lane_y)
right_lane_x = np.array(right_lane_x)
right_lane_y = np.array(right_lane_y)

coeff_left = np.polyfit(left_lane_x, left_lane_y, 2)
coeff_right = np.polyfit(right_lane_x, right_lane_y, 2)

poly_left = np.poly1d(coeff_left)
poly_right = np.poly1d(coeff_right)

scale_factor = 50
# 피팅된 곡선 그리기
for idx in range(len(left_lane_x) - 1):
    x1 = left_lane_x[idx]
    x2 = left_lane_x[idx + 1]
    y1 = int(poly_left(x1))
    y2 = int(poly_left(x2))
    cv2.line(line, (x1, y1), (x2, y2), (255, 255, 0), 2)  # 초록색 선


for idx in range(len(right_lane_x) - 1):
    x1 = right_lane_x[idx]
    x2 = right_lane_x[idx + 1]
    y1 = int(poly_right(x1))
    y2 = int(poly_right(x2))

    print(x1, x2, y1, y2)
    cv2.line(line, (x1, y1), (x2, y2), (255, 255, 0), 2)  # 초록색 선

cv2.imshow('line', line)

cv2.imshow('left_line', left_line)
cv2.imshow('right_line', right_line)

cv2.waitKey(0)
cv2.destroyAllWindows()

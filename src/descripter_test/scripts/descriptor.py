import cv2
import numpy as np

# 라바콘의 경우 이런 식으로 구분 가능함
# 그런데 자전거 탄 사람은 어떻게?
# 그리고 영상보면 다른 자동차도 인식해야 함

# 이미지 로드 (여기서는 예시 이미지로 로드)
image = cv2.imread('cone2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

o_lo = np.array([0, 190, 30])
o_hi = np.array([15, 255, 255])

r_lo = np.array([0, 190, 100])
r_hi = np.array([15, 255, 255])

o_mask = cv2.inRange(hsv, o_lo, o_hi)
r_mask = cv2.inRange(hsv, r_lo, r_hi)

# 비스무리한 빨간 영역(신호등 적색 신호)은 제거하고, 주황 영역만 구하는 코드
rev_red = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(r_mask))
orange = cv2.bitwise_and(rev_red, rev_red, mask=o_mask)

# 찌꺼지 제거 후 빈칸 채우기 - 필요하지 않을 수도 있음
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
erode = cv2.morphologyEx(orange, cv2.MORPH_ERODE, k)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
cone = cv2.morphologyEx(erode, cv2.MORPH_DILATE, k)

cone[cone > 0] = 255
area_idx = np.where(cone > 0)
y1, y2 = np.min(area_idx[0]), np.max(area_idx[0])
x1, x2 = np.min(area_idx[1]), np.max(area_idx[1])

y1 = max(y1 - 10, 0)
y2 = min(y2 + 30, image.shape[0])

x1 = max(x1 - 40, 0)
x2 = min(x2 + 35, image.shape[1])

roi = image[y1:y2, x1:x2]
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('roi', roi)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

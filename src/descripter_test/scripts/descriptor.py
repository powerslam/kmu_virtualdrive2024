import cv2
import numpy as np

# 라바콘의 경우 이런 식으로 구분 가능함
# 그런데 자전거 탄 사람은 어떻게?
# 그리고 영상보면 다른 자동차도 인식해야 함
PATH = './img/'
src = cv2.imread(PATH + 'cone4.png')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

o_lo = np.array([0, 60, 60])
o_hi = np.array([20, 255, 255])

o_img = cv2.inRange(hsv, o_lo, o_hi)

cv2.imshow('img', src)
cv2.imshow('orange', o_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# orb = cv2.ORB_create(
#     nfeatures=1000,
#     scaleFactor=2,
#     nlevels=8,
#     edgeThreshold=31,
#     firstLevel=0,
#     WTA_K=2,
#     scoreType=cv2.ORB_FAST_SCORE,
#     patchSize=31,
#     fastThreshold=20,
# )

# kp1, des1 = orb.detectAndCompute(gray, None)
# kp2, des2 = orb.detectAndCompute(target, None)

# # 해밍 거리를 가지고 비교
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)

# for i in matches[:100]:
#     idx = i.queryIdx
#     x1, y1 = kp1[idx].pt
#     cv2.circle(src, (int(x1), int(y1)), 3, (255, 0, 0), 1)

# cv2.imshow("src", src)
# cv2.imshow('target', gray)
# cv2.imshow('target', target)
# cv2.waitKey()
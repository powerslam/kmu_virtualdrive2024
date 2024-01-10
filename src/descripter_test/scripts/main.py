#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage

PATH = './img/'

# 러버콘은 카메로 찍었을 때 종료되는 버그가 존재함
class DescriptorTest():
    def __init__(self):
        rospy.init_node('descriptor_test') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.func)

        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    # 근데 아마 장애물 종류에 따라서 함수가 바뀌어야 함!!! 이게 좀 큰 문제임
    def func(self, msg: CompressedImage):
        self.img = self.get_image(msg)
        cx1, cx2, cy1, cy2 = self.find_cone()
        
        cone = self.img[cy1:cy2, cx1:cx2]

        cv2.rectangle(self.img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.imshow('image', self.img)
        cv2.imshow('cone', cone)
        
        cv2.waitKey(1)

    def find_cone(self):
        h, w, _ = self.img.shape

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 주황색 영역에 대한 색상범위
        o_lo = np.array([0, 190, 30])
        o_hi = np.array([15, 255, 255])

        # 빨강색 영역에 대한 색상범위
        r_lo = np.array([0, 190, 100])
        r_hi = np.array([15, 255, 255])

        # 색상범위에 대한 mask 값
        o_mask = cv2.inRange(hsv, o_lo, o_hi)
        r_mask = cv2.inRange(hsv, r_lo, r_hi)

        # 비스무리한 빨간 영역(신호등 적색 신호)은 제거하고, 주황 영역만 구하는 코드
        rev_red = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(r_mask))
        cone = cv2.bitwise_and(rev_red, rev_red, mask=o_mask)

        # 찌꺼지 제거 후 빈칸 채우기 - 필요하지 않을 수도 있음
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erode = cv2.morphologyEx(cone, cv2.MORPH_ERODE, k)
        
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        cone = cv2.morphologyEx(erode, cv2.MORPH_DILATE, k) # 최종 출력

        cone[cone > 0] = 255
        area_idx = np.where(cone > 0)
        y1, y2 = np.min(area_idx[0]), np.max(area_idx[0])
        x1, x2 = np.min(area_idx[1]), np.max(area_idx[1])

        y1 = max(y1 - 10, 0)
        y2 = min(y2 + 30, h)

        x1 = max(x1 - 40, 0)
        x2 = min(x2 + 35, w)

        return x1, x2, y1, y2

if __name__ == '__main__':
    try:
        pub = DescriptorTest()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


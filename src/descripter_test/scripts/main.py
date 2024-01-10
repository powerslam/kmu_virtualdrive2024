#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage

# 러버콘은 카메로 찍었을 때 종료되는 버그가 존재함
# 매번 색상값을 가지고 찾는 것은 힘이 듬
# 그러니 객체 추적 알고리즘 등을 활용하는 것이 나아 보임
# 그리고 아직 단일 객체만 판단하는 코드이기도 함

class DescriptorTest():
    def __init__(self):
        rospy.init_node('descriptor_test') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.func)

        self.img, self.hsv, self.gray = None, None, None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    # 근데 아마 장애물 종류에 따라서 함수가 바뀌어야 함!!! 이게 좀 큰 문제임
    def func(self, msg: CompressedImage):
        self.img = self.get_image(msg)
        cx1, cx2, cy1, cy2 = self.find_cone()
        
        if not (cx1 == -1 and cx2 == -1 and cy1 == -1 and cy2 == -1):
            cone = self.img[cy1:cy2, cx1:cx2]
            cv2.rectangle(self.img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.imshow('cone', cone)
                
        cv2.imshow('image', self.img)
        cv2.waitKey(1)

    # 공통 수정 사항
        # 영역 크기에 따른 bbox 비율 조정
        # 거리 별로 체크하기 
    def find_cone(self):
        # 주황색 영역에 대한 색상범위
        o_lo = np.array([0, 180, 30])
        o_hi = np.array([15, 255, 255])

        # 빨강색 영역에 대한 색상범위
        # 이 부분이 색이 밝아지면 너무 똑같아 짐
        r_lo = np.array([0, 190, 100])
        r_hi = np.array([0, 255, 255])

        # 색상범위에 대한 mask 값
        o_mask = cv2.inRange(self.hsv, o_lo, o_hi)
        cv2.imshow('o_mask', o_mask)
        
        # 빨간색 영역은 라바콘의 밝은 영역이랑 겹치는 경우가 있어서
        # 일단 다른 방법 찾기 전엔 스킵
        # r_mask = cv2.inRange(hsv, r_lo, r_hi)
        # cv2.imshow('r_mask', r_mask)

        # 비스무리한 빨간 영역(신호등 적색 신호)은 제거하고, 주황 영역만 구하는 코드 -> 일단 스킵
        # rev_red = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(r_mask))
        # cone = cv2.bitwise_and(rev_red, rev_red, mask=o_mask)
        cone = cv2.bitwise_and(self.gray, self.gray, mask=o_mask)

        # 후추 제거 후 빈칸 채우기 - 필요하지 않을 수도 있음
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erode = cv2.morphologyEx(cone, cv2.MORPH_ERODE, k)
        
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        cone = cv2.morphologyEx(erode, cv2.MORPH_DILATE, k) # 최종 출력

        # 단일 객체인 게 문제
        # 그 부분은 라이다랑 데이터르 합치면 쉽게 풀릴 것 같기도
        cone[cone > 0] = 255
        area_indices = np.where(cone > 0)
        if len(area_indices[0]):
            h, w, _ = self.img.shape

            y1, y2 = np.min(area_indices[0]), np.max(area_indices[0])
            x1, x2 = np.min(area_indices[1]), np.max(area_indices[1])

            y1 = max(y1 - 10, 0)
            y2 = min(y2 + 30, h)

            x1 = max(x1 - 40, 0)
            x2 = min(x2 + 35, w)

            return x1, x2, y1, y2
        
        return -1, -1, -1, -1

    # 사람은 바지색 보고 판단하기
    def find_person(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 남색 영역(바지)에 대한 색상범위
        # 90, 110, 160, 0, 255
        b_lo = np.array([90, 160, 0])
        b_hi = np.array([110, 255, 255])

        # 색상범위에 대한 mask 값
        b_mask = cv2.inRange(hsv, b_lo, b_hi)
        cv2.imshow('b_mask', b_mask)
        
        person = cv2.bitwise_and(gray, gray, mask=b_mask)

        person[person > 0] = 255
        area_indices = np.where(person > 0)
        if len(area_indices[0]):
            h, w, _ = self.img.shape

            y1, y2 = np.min(area_indices[0]), np.max(area_indices[0])
            x1, x2 = np.min(area_indices[1]), np.max(area_indices[1])

            # 확인 후 수정
            y1 = max(y1 - 10, 0)
            y2 = min(y2 + 30, h)

            x1 = max(x1 - 40, 0)
            x2 = min(x2 + 35, w)

            return x1, x2, y1, y2
        
        return -1, -1, -1, -1

if __name__ == '__main__':
    try:
        pub = DescriptorTest()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
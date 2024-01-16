#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 초점거리는 0으로 생각해도 됨

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class BevImg():
    def __init__(self):
        rospy.init_node('cam_sub') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback)
        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def callback(self, msg: CompressedImage) -> None:
        self.img = self.get_image(msg)
        self.bev_img = self.bev()
        #self.sliding_window()

        cv2.imshow('img', self.img)
        cv2.imshow('res', self.bev_img)
        cv2.waitKey(1)

    def bev(self):
        h, w, _ = self.img.shape
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        y_lo = np.array([15, 128, 0])
        y_hi = np.array([40, 255, 255])

        w_lo = np.array([0, 0, 200])
        w_hi = np.array([179, 64, 255])

        y_img = cv2.inRange(hsv, y_lo, y_hi)
        w_img = cv2.inRange(hsv, w_lo, w_hi)

        combined = cv2.bitwise_or(y_img, w_img)

        morph = cv2.morphologyEx(combined, cv2.MORPH_OPEN, None)

        src_pt = np.array([
            [0, 420],
            [270, 260],
            [w - 270, 260],
            [w, 420]
        ], dtype=np.float32)

        dst_pt = np.array([
            [w // 4, 480],
            [w // 4, 0],
            [w // 8 * 7, 0],
            [w // 8 * 7, 480]
        ], dtype=np.float32)

        warp = cv2.getPerspectiveTransform(src_pt, dst_pt)
        warp_img = cv2.warpPerspective(morph, warp, (w, h))

        cv2.imshow('combined', morph)

        return warp_img

    def sliding_window(self):
        self.bev_img[self.bev_img > 0] = 255
        # 255인 포인트 가져오기 ==> (y, x)
        pts = np.column_stack(np.where(self.bev_img == 255))
        mid = pts.shape[0] // 2

        left_lane = pts[mid:]
        right_lane = pts[:mid]

if __name__ == '__main__':
    try:
        pub = BevImg()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 초점거리는 0으로 생각해도 됨

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from lane_detection.msg import PixelCoord, LaneInformation

class LaneDetection():
    def __init__(self):
        rospy.init_node('lane_detection') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback)

        self.lane_info_pub = rospy.Publisher('/lane_information', LaneInformation, queue_size=5)

        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def callback(self, msg: CompressedImage) -> None:
        self.img = self.get_image(msg)
        self.h, self.w, _ = self.img.shape
        self.bev_img = self.bev()

        self.sliding_window()

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
            [235, 280],
            [w - 235, 280],
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

        warp_img = cv2.cvtColor(warp_img, cv2.COLOR_GRAY2BGR)
        warp_img[warp_img > 0] = 255

        return warp_img

    # 기울기가 무한대 == x축 y축 거리가 똑같은 경우 10000 이면 기울기가 무한대
    # ptx == (x, y)
    def get_gradient(self, pt1, pt2):
        if pt2.x - pt1.x == 0: return 10000
        
        return (pt2.y - pt1.y) / (pt2.x - pt1.x)

    def sliding_window(self):
        lane_info = LaneInformation()

        half = self.w // 2

        left_lane_candi = self.bev_img[:, :half]
        right_lane_candi = self.bev_img[:, half:]

        for height in range(self.h - 1, -1, -10):
            left_midpts = np.nonzero(left_lane_candi[height])[0]
            right_midpts = np.nonzero(right_lane_candi[height])[0] + half
            
            if len(left_midpts):
                left_midpt = PixelCoord()
                left_midpt.x = np.sum(left_midpts) // len(left_midpts)
                left_midpt.y = height

                lane_info.left_lane_points.append(left_midpt)
                cv2.circle(self.bev_img, (left_midpt.x, height), 2, (0, 255, 0), -1)

            if len(right_midpts):
                right_midpt = PixelCoord()
                right_midpt.x = np.sum(right_midpts) // len(right_midpts)
                right_midpt.y = height

                lane_info.right_lane_points.append(right_midpt)
                cv2.circle(self.bev_img, (right_midpt.x, height), 2, (0, 0, 255), -1)

        lane_info.left_gradient = -10000 if not len(lane_info.left_lane_points) else self.get_gradient(lane_info.left_lane_points[0], lane_info.left_lane_points[-1])
        lane_info.right_gradient = -10000 if not len(lane_info.right_lane_points) else self.get_gradient(lane_info.right_lane_points[0], lane_info.right_lane_points[-1])
        self.lane_info_pub.publish(lane_info)

if __name__ == '__main__':
    try:
        pub = LaneDetection()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass

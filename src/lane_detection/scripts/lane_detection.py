#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 초점거리는 0으로 생각해도 됨

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class LaneDetection():
    def __init__(self):
        rospy.init_node('cam_sub') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback)
        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def callback(self, msg: CompressedImage) -> None:
        self.img = self.get_image(msg)
        self.bev_img = cv2.cvtColor(self.bev(), cv2.COLOR_GRAY2BGR)
        self.bev_img[self.bev_img > 0] = 255
        
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

        return warp_img

    def sliding_window(self):
        win_sz = 10
        start_y = 450
        y_interval = 15
        
        lpt = [
            [180, start_y],
            [240, start_y + y_interval]
        ]

        rpt = [
            [470, start_y],
            [530, start_y + y_interval]
        ]

        for win in range(win_sz):
            self.bev_img[180:240, (start_y - y_interval * win):(start_y - y_interval * (win - 1))]

            
if __name__ == '__main__':
    try:
        pub = LaneDetection()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass

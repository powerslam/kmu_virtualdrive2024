#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Float64

from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import GetTrafficLightStatus

# 신호등은
# "SN000005" 만 가지고 동작함

# 신호 순서: 빨강 -> 초록 -> 노랑 -> 빨강+좌회전 -> 빨강+노랑 -> 빨강 -> 반복

class TrafficSub():
    def __init__(self):
        rospy.init_node('traffic_sub') 
        rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.traffic_callback)
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.chk_stop_line)
        
        self.pub_speed = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.cmd_msg = Float64()
        self.rate = rospy.Rate(2)
        self.speed = 1000

        self.stream_sign = False
        self.traffic_status = 0
        self.TRAFFIC = { 'red': 1, 'yellow': 4, 'green': 16, 'left': 32 }

        self.red_light = True

        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        self.img = None
        self.hsv = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def condition_move(self) -> None:
        if self.red_light:
            self.cmd_msg.data = 0

        else:
            self.cmd_msg.data = self.speed

        self.pub_speed.publish(self.cmd_msg)
        self.rate.sleep()

    # 신호에 따라서 self.red_light 을 바꿈
    def traffic_callback(self, msg: GetTrafficLightStatus) -> None:    
        if msg.trafficLightIndex != 'SN000005':
            self.stream_sign = False
            return
        
        self.stream_sign = True
        self.red_light = msg.trafficLightStatus != 33

    def chk_stop_line(self, msg: CompressedImage) -> None:
        if not self.stream_sign:
            print("don't stream sign")
            return
        
        print('processing...')
        self.img = self.get_image(msg)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        sign_lo = np.array([0, 200, 200])
        sign_hi = np.array([180, 255, 255])

        sign = cv2.inRange(self.hsv, sign_lo, sign_hi)
        sign = cv2.morphologyEx(sign, cv2.MORPH_DILATE, self.kernel3)

        w_lo = np.array([0, 0, 200])
        w_hi = np.array([180, 64, 255])

        white = cv2.inRange(self.hsv, w_lo, w_hi)
        white = cv2.morphologyEx(white, cv2.MORPH_DILATE, self.kernel5)

        bev = self.bev_transform(white)

        cv2.imshow('sign', sign)
        cv2.imshow('bev', bev)
        cv2.waitKey(1)

    def bev_transform(self, img: np.ndarray):
        h, w, _ = self.img.shape

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
        warp_img = cv2.warpPerspective(img, warp, (w, h))
        return warp_img

if __name__ == '__main__':
    try:
        pub = TrafficSub()
        while not rospy.is_shutdown():
            pub.condition_move()        
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


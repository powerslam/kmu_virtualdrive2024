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

        # 정지할 때 카메라랑 속도 맞춰야 함
        self.rate = rospy.Rate(30)

        # 차량 속도에 따라 체크해야 하는 영역이 다름
        self.speed = 1000

        self.stream_sign = False
        self.traffic_status = 0
        self.TRAFFIC = { 'red': 1, 'yellow': 4, 'green': 16, 'left': 32 }

        self.red_light = True

        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        self.img = None
        self.hsv = None

        self.stop_line_mask = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.stop_line_mask[350:430, 40:520] = 255

        self.stop = False
        self.flag = True

        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def condition_move(self) -> None:
        if self.stop:
            self.cmd_msg.data = 0

        else:
            self.cmd_msg.data = self.speed

        # print(self.cmd_msg)
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
        self.img = self.get_image(msg)

        bev = self.bev_transform()
        res = len(cv2.bitwise_and(bev, self.stop_line_mask).nonzero()[0])
        
        if self.flag:
            self.stop = res > 80000
        
        if self.stop:
            self.flag = False   

        cv2.imshow('bev', bev)
        cv2.waitKey(1)

    def bev_transform(self):
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

if __name__ == '__main__':
    try:
        pub = TrafficSub()
        while not rospy.is_shutdown():
            pub.condition_move()
            pub.rate.sleep()

        rospy.spin()

    except rospy.ROSInterruptException:
        pass


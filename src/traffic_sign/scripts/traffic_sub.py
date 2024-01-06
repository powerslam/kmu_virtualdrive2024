#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Float64

from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import GetTrafficLightStatus

# 신호 순서: 빨강 -> 초록 -> 노랑 -> 빨강+좌회전 -> 빨강+노랑 -> 빨강 -> 반복

# 로직 
# 정지선이 나타날 때까지 주행
# if 정지선이 나타남
#    다시 사라질 때까지 주행? ==> 바로 앞에서 멈출 수 있도록 주행
#    신호에 맞추어 정지 or 다음 코스 이동

class TrafficSub():
    def __init__(self):
        rospy.init_node('traffic_sub') 
        rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.traffic_callback)
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.chk_stop_line)
        
        self.pub_speed = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.cmd_msg = Float64()
        self.rate = rospy.Rate(2)
        self.speed = 1000
        
        self.traffic_status = 0
        self.TRAFFIC = { 'red': 1, 'yellow': 4, 'green': 16, 'left': 32 }

        self.can_go_line = True
        self.can_go_traffic = True

        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def condition_move(self) -> None:
        print('line: ', self.can_go_line, 'traffic: ', self.can_go_traffic)

        # 정지선 바로 앞이지만, 신호가 괜찮으면 출발
        # 정지선 바로 앞이고, 신호가 안 괜찮으면 정지
        if not self.can_go_line and not self.can_go_traffic:
            self.cmd_msg.data = 0

        else:
            self.cmd_msg.data = self.speed

        self.pub_speed.publish(self.cmd_msg)
        self.rate.sleep()

    # 신호에 따라서 self.can_go_traffic 을 바꿈
    def traffic_callback(self, msg: GetTrafficLightStatus) -> None:
        traffic_status = msg.trafficLightStatus
        traffic_status ^= self.TRAFFIC['red']
        traffic_status ^= self.TRAFFIC['yellow']

        # 조건을 잘 찾아봐야 할 듯
        # 신호등 앞 정지선 부근에서? 신호가 수신됨
        if traffic_status < 6:
            self.can_go_traffic = False
            #print(msg.header.stamp, 'stop!!')
            pass
        elif traffic_status == self.TRAFFIC['yellow'] + self.TRAFFIC['left']:
            self.can_go_traffic = True
            #print(msg.header.stamp, 'go left')
            pass
        else:
            self.can_go_traffic = True
            #print(msg.header.stamp, 'go')
            pass

    # HoughLinesP 를 활용해서 가로선을 구함
    # self.can_go_line을 바꿈 ==> 만약에 정지선이 430 이하면?
    def chk_stop_line(self, msg: CompressedImage) -> None:
        # y 축 평균을 구하기 위해 x 값 별로 y 평균을 구함
        self.img = self.get_image(msg)
        self.bev = self.bev_transform()
        
        self.bev[self.bev > 0] = 255

        lines = cv2.HoughLinesP(self.bev, 1, np.pi / 180., 200, minLineLength=350, maxLineGap=1)
        dst = cv2.cvtColor(self.bev, cv2.COLOR_GRAY2BGR)

        # 라인이 존재하면
        res_pt = None
        if lines is not None:
            for i in range(lines.shape[0]):
                pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표 x,y
                pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표, 가운데는 무조건 0
                
                if pt2[0] - pt1[0] == 0: 
                    continue
                
                if abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0])) >= 0.7:
                    continue
                
                if res_pt is not None:
                    if abs((res_pt[2] - res_pt[0]) / (res_pt[3] - res_pt[1])) < abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0])):
                        res_pt = lines[i][0]

                else:
                    res_pt = lines[i][0]

            if res_pt is not None:
                self.can_go_line = (res_pt[1] + res_pt[3]) // 2 <= 410
                cv2.line(dst, (res_pt[0], res_pt[1]), (res_pt[2], res_pt[3]), (0, 0, 255), 2, cv2.LINE_AA)
        
        # 라인이 존재하지 않을 때
        if res_pt is None:
            # print('line has gone')
            self.can_go_line = True
        
        cv2.imshow('dst', dst)
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

if __name__ == '__main__':
    try:
        pub = TrafficSub()
        while not rospy.is_shutdown():
            pub.condition_move()        
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


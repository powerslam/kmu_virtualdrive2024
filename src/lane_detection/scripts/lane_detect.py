#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np

from std_msgs.msg import Int32

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from lane_detection.msg import PixelCoord
from lane_detection.msg import LaneInformation

class LaneDetection():
    def __init__(self):
        rospy.init_node('lane_detect')
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.callback)

        self.lane_info_pub = rospy.Publisher('/lane_information', LaneInformation, queue_size=5)
        self.stop_lane_pub = rospy.Publisher('/stop_lane_information', Int32, queue_size=5)

        self.img = None
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

        self.start_left_x = -1
        self.start_right_x = -1

        self.stop_lane_mask = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.stop_lane_mask[250:440, 40:520] = 255

    def callback(self, msg: CompressedImage) -> None:
        self.img = self.get_image(msg)
        self.h, self.w, _ = self.img.shape
        self.bev_img = self.bev()
        self.bev_img_draw = self.bev()

        self.sliding_window()

        # cv2.imshow('img', self.img)
        cv2.imshow('res', self.bev_img_draw)
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
            [240, 280],
            [w - 240, 280],
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
        # 만약에 start를 하지도 못한 경우
        if pt2[0] == -1 or pt1[0] == 0: return 480.
        if pt2[0] - pt1[0] == 0: return 480.
        return (pt1[1] - pt2[1]) / (pt2[0] - pt1[0])

    def sliding_window(self):
        lane_info = LaneInformation()

        nonzero = self.bev_img[:,:,0].nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        start_left_lane = self.bev_img[:,:,0][430:480, :342].nonzero()[1]
        start_right_lane = self.bev_img[:,:,0][430:480, 342:].nonzero()[1]

        pub_grad_left_s_x = -1
        pub_grad_left_s_y = -1
        
        pub_grad_right_s_x = -1
        pub_grad_right_s_y = -1
        
        pub_grad_left_e_x = 0
        pub_grad_left_e_y = 0
        
        pub_grad_right_e_x = 0
        pub_grad_right_e_y = 0
        
        if len(start_left_lane) == 0:
            if self.start_left_x == -1:
                self.start_left_x = 190

            left_midpt_x = self.start_left_x

        else:
            left_midpt_x = int(start_left_lane.mean())
            self.start_left_x = left_midpt_x
            
            pub_grad_left_s_x = left_midpt_x
            pub_grad_left_s_y = 0

        if len(start_right_lane) == 0:
            if self.start_right_x == -1:
                self.start_right_x = 500

            right_midpt_x = self.start_right_x

        else:
            right_midpt_x = int(start_right_lane[-30:].mean()) + 325
            self.start_right_x = right_midpt_x

            pub_grad_right_s_x = right_midpt_x
            pub_grad_right_s_y = 0

        window = 20
        h_interval = 24
        x_padding = 60

        left_gradient = 0
        right_gradient = 0

        prev_left_x = None
        prev_right_x = None

        # y 값은 480, 430, 380, ....
        for win in range(window):
            left_lane_pt = PixelCoord()
            right_lane_pt = PixelCoord()

            y_min = 480 - (win + 1) * h_interval
            y_max = 480 - win * h_interval

            mid_y = (y_min + y_max) // 2

            left_x_min = left_midpt_x - x_padding
            left_x_max = left_midpt_x + x_padding
            cv2.rectangle(self.bev_img_draw, (left_x_min, y_min), (left_x_max, y_max), (255, 0, 0), 1, cv2.LINE_AA)

            good_left_inds = ((nonzeroy >= y_min) & (nonzeroy < y_max) & (nonzerox >= left_x_min) & (nonzerox < left_x_max)).nonzero()[0]
            if len(good_left_inds) > 50:
                left_midpt_x = int(nonzerox[good_left_inds].mean())        
                left_lane_pt.x = left_midpt_x

                # None이 아닌 경우
                if prev_left_x:
                    left_gradient = left_midpt_x - prev_left_x

                    # 만약에 grad가 시작 안 했으면
                    if pub_grad_left_s_x == -1:
                        pub_grad_left_s_x = left_midpt_x
                        pub_grad_left_s_y = win

                pub_grad_left_e_x = left_midpt_x
                pub_grad_left_e_y = win
                    
            else:
                left_lane_pt.x = left_midpt_x + left_gradient
                left_midpt_x = left_midpt_x + left_gradient
            
            prev_left_x = left_lane_pt.x

            left_lane_pt.y = win
            cv2.circle(self.bev_img_draw, (left_midpt_x, mid_y), 2, (0, 255, 0), -1)

            right_x_min = right_midpt_x - x_padding
            right_x_max = right_midpt_x + x_padding

            cv2.rectangle(self.bev_img_draw, (right_x_min, y_min), (right_x_max, y_max), (0, 255, 0), 1, cv2.LINE_AA)

            good_right_inds = ((nonzeroy >= y_min) & (nonzeroy < y_max) & (nonzerox >= right_x_min) & (nonzerox < right_x_max)).nonzero()[0]
            if len(good_right_inds) > 50:
                tmp = nonzerox[good_right_inds]

                right_midpt_x = int(tmp[tmp >= np.max(tmp) - 30].mean())
                right_lane_pt.x = right_midpt_x

                # None이 아닌 경우
                if prev_right_x:
                    right_gradient = right_midpt_x - prev_right_x

                    if pub_grad_right_s_x == -1:
                        pub_grad_right_s_x = right_midpt_x
                        pub_grad_right_s_y = win

                pub_grad_right_e_x = right_midpt_x
                pub_grad_right_e_y = win

            else:
                right_lane_pt.x = right_midpt_x + right_gradient
                right_midpt_x = right_midpt_x + right_gradient

            right_lane_pt.y = win
            cv2.circle(self.bev_img_draw, (right_midpt_x, mid_y), 2, (0, 255, 0), -1)

            lane_info.left_lane_points.append(left_lane_pt)
            lane_info.right_lane_points.append(right_lane_pt)

        _common_y = ((480 - (lane_info.left_lane_points[7].y + 1) * h_interval) + (480 - lane_info.left_lane_points[7].y * h_interval)) // 2

        if lane_info.left_lane_points[7].x != 0:
            cv2.line(self.bev_img_draw, (lane_info.left_lane_points[7].x, _common_y), (190, _common_y), (0, 255, 0), 4, cv2.LINE_AA)
        
        if lane_info.right_lane_points[7].x != 0:
            cv2.line(self.bev_img_draw, (lane_info.right_lane_points[7].x, _common_y), (498, _common_y), (0, 0, 255), 4, cv2.LINE_AA)
    
        if lane_info.right_lane_points[7].x != 0 and lane_info.left_lane_points[7].x != 0:
            _mid = (lane_info.left_lane_points[7].x + lane_info.right_lane_points[7].x) // 2
            cv2.line(self.bev_img_draw, (_mid, _common_y), (342, _common_y), (255, 0, 0), 4, cv2.LINE_AA)

        
        lane_info.left_gradient = -10000 if not len(lane_info.left_lane_points) else self.get_gradient((pub_grad_left_s_x, pub_grad_left_s_y), (pub_grad_left_e_x, pub_grad_left_e_y))
        lane_info.right_gradient = -10000 if not len(lane_info.right_lane_points) else self.get_gradient((pub_grad_right_s_x, pub_grad_right_s_y), (pub_grad_right_e_x, pub_grad_right_e_y))
        
        masking = cv2.bitwise_and(self.bev_img, self.stop_lane_mask)
        self.stop_lane_pub.publish(len(masking.nonzero()[0]))
        
        self.lane_info_pub.publish(lane_info)

        # print('left', lane_info.left_gradient, 'right', lane_info.right_gradient)

if __name__ == '__main__':
    try:
        _ = LaneDetection()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass

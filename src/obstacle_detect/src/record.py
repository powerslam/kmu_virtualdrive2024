#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

# from lidar.msg import Obstacle, ObstacleArray
from sensor_msgs.msg import CompressedImage

# 러버콘은 카메로 찍었을 때 종료되는 버그가 존재함

class DescriptorTest():
    def __init__(self):
        rospy.init_node('descriptor_test') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.func)

        self.img, self.hsv, self.gray = None, None, None
        self.obstacle_info = None
        
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

        self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480)) # 파일명, 코덱, FPS, 해상도
    
        self.pixel_interval = 0.55 / 225

        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    # 근데 아마 장애물 종류에 따라서 함수가 바뀌어야 함!!! 이게 좀 큰 문제임
    def func(self, msg: CompressedImage):
        self.img = self.get_image(msg)
        self.out.write(self.img)

        cv2.imshow('hi', self.img)
        
        # 키보드 입력을 기다리고 종료할 수 있도록 함
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.release_resources()
            rospy.signal_shutdown("User terminated")
    
    def release_resources(self):
        cv2.destroyAllWindows()
        self.out.release()

if __name__ == '__main__':
    try:
        pub = DescriptorTest()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
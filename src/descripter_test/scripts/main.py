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
        
        self.orb = cv2.ORB_create(
            nfeatures=40000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

        self.img = None
        self.cone = cv2.imread(PATH + 'cone4.png')
        self.kp1, self.des1 = self.orb.detectAndCompute(self.cone, None)

        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

    def func(self, msg: CompressedImage):
        self.img = self.get_image(msg)
        self.matching()

        cv2.imshow('img', self.img)
        cv2.waitKey(1)

    def matching(self):
        _, des2 = self.orb.detectAndCompute(self.img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        for i in matches[:100]:
            idx = i.queryIdx
            x1, y1 = self.kp1[idx].pt
            cv2.circle(self.img, (int(x1), int(y1)), 3, (255, 0, 0), 3)

if __name__ == '__main__':
    try:
        pub = DescriptorTest()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


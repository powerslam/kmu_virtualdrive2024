#!/usr/bin/env python3
#-*-coding:utf-8-*-

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge

from lidar.msg import ObstacleArray
from sensor_msgs.msg import CompressedImage

from obstacle_detect.msg import DynamicObstacleInfo, DynamicObstacleInfoArray

def rotation_from_euler(roll=1., pitch=1., yaw=1.):
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R

def translation_matrix(vector):
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M

class DescriptorTest():
    def __init__(self):
        rospy.init_node('descriptor_test') 
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.camera_obstacle_callback)
        rospy.Subscriber('/obstacle_info', ObstacleArray, self.lidar_obstacle_callback)

        self.dynamic_obstacle_info_array_pub = rospy.Publisher('/dynamic_obstacle_information', DynamicObstacleInfoArray, queue_size=10)

        self.img, self.hsv, self.gray = None, None, None
        self.obstacle_info = None
        
        self.bridge = CvBridge()
        self.get_image = lambda msg: self.bridge.compressed_imgmsg_to_cv2(msg)

        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        fx, fy = 320., 320.
        u0, v0 = 320., 240.

        intrinsic = np.array([[fx, 0., u0, 0.],
                              [0., fy, v0, 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]], dtype=np.float32)

        R = np.array([[0., 1., 0., 0.],
                      [0., 0., -1., 0.],
                      [-1., 0., 0., 0.],
                      [0., 0., 0., 1.]], dtype=np.float32)
        
        roll, pitch, yaw = 0., 0., 0.
        x, y, z = 0.19, 0., -0.02

        R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
        T_veh2cam = translation_matrix((-x, -y, -z))

        extrinsic = R @ R_veh2cam @ T_veh2cam

        self.ipm_matrix = intrinsic @ extrinsic 

    def lidar_obstacle_callback(self, msg: ObstacleArray):
        infos = msg.obstacle_infos
        self.obstacle_info = np.array([[info.obst_y, -info.obst_x, 0., 1.] for info in infos])

    def camera_obstacle_callback(self, msg: CompressedImage):
        self.img = self.get_image(msg)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        
        data = DynamicObstacleInfoArray(isObstacle=False)
        if self.obstacle_info is not None and len(self.obstacle_info) != 0:
            image_coords = self.ipm_matrix @ self.obstacle_info.T
            image_coords /= image_coords[2]

            uv = image_coords[:2, :].T
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= 640)]
            
            # print('uv', uv)

            for idx, info in enumerate(uv):
                data.isObstacle = True
                x, y = map(int, info)
                
                cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)

                res = self.find_person(x, y)
                if res:
                    _x = -self.obstacle_info[idx][1]
                    _y = self.obstacle_info[idx][0]

                    data.dynamic_obstacles.append(DynamicObstacleInfo(x=_x, y=_y, distance=np.sqrt(_x ** 2 + _y ** 2)))
                    cv2.rectangle(self.img, (x - 20, y - 30), (x + 20, y + 10), (0, 255, 0), 2)

                else:
                    cv2.rectangle(self.img, (x - 20, y - 30), (x + 20, y + 10), (255, 0, 0), 2)

        self.dynamic_obstacle_info_array_pub.publish(data)

        cv2.imshow('image', self.img)
        cv2.waitKey(1)

    # 사람은 바지색 보고 판단하기
    def find_person(self, x, y):
        # 남색 영역(바지)에 대한 색상범위
        # 90, 110, 160, 0, 255
        b_lo = np.array([90, 160, 0])
        b_hi = np.array([110, 255, 255])

        # 색상범위에 대한 mask 값
        b_mask = cv2.inRange(self.hsv, b_lo, b_hi)
        # cv2.imshow('b_mask', b_mask) # DEBUG
        
        person = cv2.bitwise_and(self.gray, self.gray, mask=b_mask)
        # 후추 제거 후 빈칸 채우기 - 필요하지 않을 수도 있음
        erode = cv2.morphologyEx(person, cv2.MORPH_ERODE, self.kernel3)
        # cv2.imshow('erode', erode) # DEBUG
        
        person = cv2.morphologyEx(erode, cv2.MORPH_DILATE, self.kernel5) # 최종 출력
        person[person > 50] = 255

        return len(person[y - 30:y + 10, x - 20:x + 20].nonzero()[0]) > 50

if __name__ == '__main__':
    try:
        pub = DescriptorTest()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
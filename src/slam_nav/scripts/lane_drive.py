#! /usr/bin/env python3

import tf
import rospy
import pickle
import actionlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt

import os

from nav_msgs.msg import Odometry, Path
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist, Pose

from lane_detection.msg import LaneInformation, PixelCoord

# 신호등 토픽
from morai_msgs.msg import GetTrafficLightStatus

# 카메라
from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Float64, Int32

# 카메라 정보
from obstacle_detect.msg import ObstacleInfo, ObstacleInfoArray

# 라이다 정보
from obstacle_detect.msg import LidarObstacleInfo, LidarObstacleInfoArray, RotaryArray
from actionlib_msgs.msg import GoalStatus

class Total:
    def __init__(self):
        rospy.init_node("navigation_client")

        self.vel_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        
        self.stop_data = Float64(data=0)

        self.target_vel = 1200
        self.L_curv, self.R_curv, self.L_point, self.R_point = 0, 0, [], []
        rospy.Subscriber("/lane_information", LaneInformation, self.lane_callback)

        self.prev_stop_lane_flag = False
        self.stop_lane_flag = False
    
    def lane_callback(self, msg: LaneInformation):
        self.L_curv = msg.left_gradient
        self.R_curv = msg.right_gradient
        self.L_point = msg.left_lane_points
        self.R_point = msg.right_lane_points

        # 0.07 ~ 0.25 언저리가 나옴
        # 0.07 일때 최대 조향 ==> 좌회전일 때 양수
        # 좌회전인 경우
        self.steer_angle = 0.5
        # if 0 < self.R_curv < 0.1:
        # self.target_vel = 2400

        # 기준이 
        some_value = 1 / (self.R_curv * 2.5) # 살짝 키운값

        MID_POINT = 342

        l7_pt = self.L_point[7].x
        r7_pt = self.R_point[7].x

        m = (l7_pt + r7_pt) // 2

        # 1차 조향각
        self.steer_angle = self.mapping(some_value, -20, 20, 1, 0)

        # 오른차선의 중앙값보다 r7_pt 의 좌표가 작으면 => 왼쪽으로 더 틀어야 함
        # 오른차선의 중앙값보다 r7_pt 의 좌표가 크면 => 오른쪽으로 더 틀어야 함
        self.steer_angle -= (MID_POINT - m) / 1000

        print(self.steer_angle)

            # self.mapping(self.R_curv, 0.07, 0.25, 0.5, 0)

        # 일단 amcl_pose 로 이동하다가
        # grad가 일정 이하로 떨어지면
        # 그 때 amcl_pose 갱신?

        # 그리고 grad가 일정 이하로 떨어진 그 때 한 순간은
        # 다시 grad가 복구될 때까지 "차선으로만" 주행
        
        # 복구가 된다면 조정된 amcl_pose 가지고 주행

        # 이걸 반복

        # amcl_pose 수정하잖아
        # sequence 이런거 

        # 기본적으로 오른쪽 차선에 대한 값만 가지고 주행을 하는 걸로
        # 그러다가 정지선을 뭐 4번 봤어 그 상태에서 왼쪽 차선이 사라지면?
        # 왼쪽 차선이 보일때까지 왼쪽으로 꺾고
        # 로터리 전까지 움직임
        self.run()

    def create_trajectory(self, tgt_x, tgt_y, dist):
        x = np.array([self.now_pose.x, self.now_pose.x + dist * 0.3, tgt_x - dist * 0.3, tgt_x])
        y = np.array([self.now_pose.y, self.now_pose.y       , tgt_y       , tgt_y])

        #print('trajectory x', x)
        #print('trajectory y', y)

        x_scale = x
        y_scale = y

        f = CubicHermiteSpline(x_scale, y_scale, dydx = [0, 0, 0, 0])

        coefficients = f.c.T # 생성된 Cubic의 계수 (coefficients)

        num = int(dist / 0.05) 
        x_new = np.linspace(x_scale[0], x_scale[-1], num) # 경로 상의 절대 x 좌표
        y_new = f(x_new)  #경로상의 절대 y좌표

        orientation_new = []  #경로상의 orientation

        for i in range(len(x_new)):
            if x_new[i] < x[1]:
                a, b, c, d = coefficients[0]
                # y_new.append(a*(x_new[i] - x[0])**3 + b*(x_new[i] - x[0])**2 + c*(x_new[i] - x[0]) +d)
                orientation_new.append(\
                    3 * a * (x_new[i] - x[0]) ** 2 + 2 * b * (x_new[i] - x[0]) + c\
                )

            elif x_new[i] < x[2]:
                a, b, c, d = coefficients[1]
                # y_new.append(a*(x_new[i] - x[0])**3 + b*(x_new[i] - x[0])**2 + c*(x_new[i] - x[0]) +d)
                orientation_new.append( 3 * a * (x_new[i] - x[0]) ** 2 + 2 * b * (x_new[i] - x[0]) + c)
            
            else: 
                a, b, c, d = coefficients[2]
                #y_new.append(a*(x_new[i] - x[0])**3 + b*(x_new[i] - x[0])**2 + c*(x_new[i] - x[0]) +d)
                orientation_new.append( 3 * a * (x_new[i] - x[0]) ** 2 + 2 * b * (x_new[i] - x[0]) + c)

        # #print(orientation_new)

        self.obstacle_point = 0
        for seq in range(self.sequence + 1, self.sequence + num):
            self.obstacle_point += 1
            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.position.x = x_new[self.obstacle_point]
            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.position.y = y_new[self.obstacle_point]
            _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, orientation_new[self.obstacle_point])

            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.orientation.x = 0.
            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.orientation.y = 0.
            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.orientation.z = qz
            self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.orientation.w = qw
        
        #print("회피경로 생성 완료 ")

        plt.figure(figsize = (dist, 0.5))
        plt.plot(x_new, y_new, 'b')
        plt.plot(x_scale, y_scale, 'ro')
        plt.plot(x_new, orientation_new, 'g')
        
        plt.title('Cubic Hermite Spline Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def run(self):
        self.vel_pub.publish(self.target_vel)
        self.steer_pub.publish(self.steer_angle)

    def mapping(self, value, from_min, from_max, to_min, to_max):
        value = np.clip(value, from_min, from_max)

        # 선형 변환 수행
        from_range = from_max - from_min
        to_range = to_max - to_min

        # 선형 변환 공식 적용
        mapped_value = ((value - from_min) / from_range) * to_range + to_min

        return mapped_value

    def normalize_angle(self, angle):
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        #angle = (angle / (np.pi))*0.5
        return angle

    def direction_vector(self, yaw):
        return np.array([np.cos(yaw), np.sin(yaw)])

    def stop(self):
        self.vel_pub.publish(Float64(data=0))  # 속도를 0으로 설정하여 정지

if __name__ == "__main__":
    nc = Total()
    rospy.spin()
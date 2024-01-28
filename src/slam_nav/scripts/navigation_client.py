#! /usr/bin/env python3

import rospy
import pickle
from math import *
import numpy as np

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lane_detection.msg import LaneInformation, PixelCoord
from nav_msgs.msg import Odometry  # /odometry/filtered 토픽 수신
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
import tf


class pidControl:
    def __init__(self):
        self.p_gain = 0
        self.i_gain = 0
        self.d_gain = 0
        self.prev_error = 0
        self.i_control = 0
        self.controllTime = 0.0333

    def pid(self, target_vel, current_vel):
        error = target_vel - current_vel

        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controllTime
        d_control = self.d_gain * (error - self.prev_error) / self.controllTime

        output = (p_control + self.i_control + d_control) + current_vel
        self.prev_error = error
        return output


class NavigationClient:
    def __init__(self):
        rospy.init_node("navigation_client")
        self.now_pose = None

        self.vel_pub = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=1
        )  # linear.x (진행방향 속도), angular.z(회전 각도)
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.client.wait_for_server()
        self.goal_list = []
        self.sequence = 0  # 받아온 좌표모음의 index

        with open(
            "/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl", "rb"
        ) as file:
            pt_list = pickle.load(file)

            for _pt in pt_list:
                pt = MoveBaseGoal()
                pt.target_pose.header.frame_id = "map"
                pt.target_pose.pose.position.x = _pt.position.x
                pt.target_pose.pose.position.y = _pt.position.y
                pt.target_pose.pose.orientation.z = _pt.orientation.z
                pt.target_pose.pose.orientation.w = _pt.orientation.w

                self.goal_list.append(pt)

        self.goal_list.extend(self.goal_list[::-1])  # 받아온 목표 좌표 모음

        self.vehicle_length = 0.26  # 차량 길이 설정
        # self.lookahead_distance = 0.8085 # Lookahead distance 설정
        # self.lookahead_distance = 0.3 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        # self.lookahead_distance = 0.598 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        self.lookahead_distance = (
            0.8  # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =0.8 일때( 3km/h) 0.35
        )
        self.angle_offset = 0
        self.gain = 1.0

        self.is_current_vel = False
        self.is_state = False
        self.target_vel = 0
        self.current_vel = 0.0

        self.L_curv = 0
        self.R_curv = 0
        self.L_point = 0
        self.R_point = 0

        self.NO_RIGHTLINE=False
        self.NO_LEFTLINE=False 

        self.pid_controller = pidControl()
        self.start_time = rospy.Time.now()

        self.CURVE_ENDPOINT = 20

        self.curve_endpoint = self.CURVE_ENDPOINT # 코너 끝까지 빠져나올 때까지 속도 유지하기 위함
        # self.curve_endpoint = True
        self.curve_start = True
        self.corner_count=0
        self.control_angle = Twist()

        self.dist = (
            lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2)
            ** 0.5
        )

        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback)
        rospy.Subscriber("/lane_information", LaneInformation, self.lane_callback)
        rospy.Subscriber(
            "/odometry/filtered", Odometry, self.Ld_callback
        )  # 현재 차량의 속도 받아오기

    def lane_callback(self, msg: LaneInformation):
        # 직선 = 곡률 반지름이 엄청 커지기에 곡률은 작다straight 보다 작으면
        # 곡선
        self.L_curv = msg.left_gradient
        self.R_curv = msg.right_gradient
        self.L_point = msg.left_lane_points
        self.R_point = msg.right_lane_points

        L3_point = self.L_point[3].x
        R3_point = self.R_point[3].x
        # 1/25 19:46 ==> 4 였음
        reference_quat = self.goal_list[
            self.sequence + 8
        ].target_pose.pose.orientation  # 이전 4
        reference_yaw = self.get_yaw_from_orientation(reference_quat)
        yaw = self.get_yaw_from_orientation(self.now_orientation)

        # @차선 변경 조건 각도 (30도? 40도 추가해야함)
        # print('yaw', yaw - reference_yaw)
        if abs(yaw - reference_yaw) > 0.25:  # 각도 차이가 어느정도 난다. 회전해야함
            self.target_vel = 0.8
            if self.curve_start:
                print("커브구간 진입  ")
                self.curve_start = False
            return

            # 속도 줄이기
        else:
            # print(' straight line ')
            if (
                0.3 < self.lookahead_distance < 0.77 and self.curve_endpoint
            ):  # Ld값 짧은상태=코너 주행중이었다면, 2번 속도 증가무시
                self.target_vel = 0.8
                # self.curve_endpoint = False
                self.curve_endpoint -= 1
            else:
                self.control_angle.linear.x = 2.2
                self.target_vel = 0
                # self.curve_endpoint = True
                self.curve_endpoint = self.CURVE_ENDPOINT
                # 8km/h 일 때 twist 2.0

            mid_point = (L3_point + R3_point) / 2

            mid_r_point = 498
            mid_l_point = 146

            C_straight = 1.8

            if L3_point == 0 or R3_point == 0:  # 오른쪽이나 왼쪽 차선 없음
                if L3_point == 0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                    self.angle_offset = C_straight * (mid_r_point - R3_point) / 10000
                    self.NO_LEFTLINE=True
                    self.NO_RIGHTLINE=False
                elif R3_point == 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                    self.angle_offset = C_straight * (L3_point - mid_l_point) / 10000
                    self.NO_RIGHTLINE=True 
                    self.NO_LEFTLINE=False  

                else:  # 둘다 0인 경우 > 아무래도 패쓰
                    self.angle_offset = 0
                    self.NO_LEFTLINE=self.NO_RIGHTLINE=True

            else:
                self.angle_offset = C_straight * (322 - mid_point) / 10000
                self.NO_RIGHTLINE=self.NO_LEFTLINE=False 

            """
            self.angle_offset
            회전을 해야될 때 일단 차선 보지 말자 
            직진주행일때, 양쪽의 선을 보자 
            """

    # R_curv 가 0인 경우 대책을 세워야 함
    def Ld_callback(self, data: Odometry):
        v = data.twist.twist.linear.x

        # 오른쪽 차선의 기울기
        Curv = abs(self.R_curv)

        # 회전하는 경우
        if v < 0.4:  # 맨 처음 출발할때
            self.lookahead_distance = 0.8

        elif v < 0.9:
            v = 0.8
            k_vel = 0.382
            k_curv = -0.056
            Curv = 1  # Ld 0.3

            self.lookahead_distance = 0.79
        # 직진하는 경우
            
        else:
            v = 2
            k_vel = 0.085
            k_curv = 0.021
            Curv = 34

            self.lookahead_distance = (k_vel * v) + (k_curv * Curv)

        # print(f'Ld값은 :{self.lookahead_distance}')
        self.run()

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        return euler[2]  # yaw

    def run(self):
        # print(self.sequence)
        # 장애물 감지 코드 넣고 감지됐을 경우는 send goal 실행되도록. 이때는 sequence 여러개 skip
        # 2구간의 시작 지점 sequence 이미 지났을 경우 2구간의 마지막을 goal로 잡고 teb플래너 실행시키자
        # 3구간의 시작 지점 sequence 이미 지났을 경우 구간의 마지막을 goal로 잡고 teb플래너 실행시키자

        if not self.now_pose:
            return
        if (
            self.dist(self.goal_list[self.sequence].target_pose.pose.position)
            < self.lookahead_distance
        ):
            if self.sequence >= len(self.goal_list):
                # print('end~~?')
                return
            self.sequence += 1

        dy = self.goal_list[self.sequence].target_pose.pose.position.y - self.now_pose.y
        dx = self.goal_list[self.sequence].target_pose.pose.position.x - self.now_pose.x

        # Pure Pursuit 알고리즘 적용
        angle_to_target = atan2(dy, dx)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        angle_difference = angle_to_target - yaw
        angle_difference = self.normalize_angle(angle_difference)
        # print(angle_difference)
        # 조향각 계산

        angle_diff_size = abs(angle_difference)
        test = abs(angle_difference)

        corner_gain_min = 4.93

        # 이 친구가 왔다리 갔다리
        if self.target_vel == 0.8:
            test_test = 1.0 + test * 0.99

            test_test = np.clip(test_test, corner_gain_min, 10)

            print(f"곡선에서 gain값: {test_test}")
            self.corner_count += 1
            self.gain = test_test
            
        else:
            if test < 0.04:
                test_test = 1.0 + test * 1.4
                test_test = np.clip(test_test, 1.0, 2.0)
                print(f"똑바른 직선에서 gain값: {test_test}")
                self.corner_count = 0
            
            else:
                if self.corner_count > 4:
                    test_test = 1.0 + test * 0.99
                    test_test = np.clip(test_test, corner_gain_min, 5.8)
                    print(f"코너 끝나고 수평 안맞을 때 gain값: {test_test}")

                else:
                    if self.NO_LEFTLINE or self.NO_RIGHTLINE: #둘 중에 하나라도 차선인식이 안되는 경우 
                        pass
                        print('차선인식 못함 ')
                    else:
                        constant_multiplier = (5 - 1.5) / (2.9 - 0.8)
                        test_test = 1.0 + test * 2
                        test_test = (test_test - 1.5) / constant_multiplier
                        test_test = np.clip(test_test, 5.0, 5.7)
                        self.target_vel = 2.0
                        print(f"직선에서 어긋났을때 gain값: {test_test}")
                    
            self.gain = test_test

        steering_angle = self.gain * atan2(
            2.0 * self.vehicle_length * sin(angle_difference) / self.lookahead_distance,
            1.0,
        )  # arctan ( 2Lsin(a)/Ld) )
        # print('steer', steering_angle, 'angle_difference', angle_difference)

        # speed=0.8
        # 조향각과 속도를 Twist 메시지로 전송
        self.control_angle = Twist()
        # self.control_angle.linear.x = speed

        # 좌회전해야 되는데 차선은 앞으로 가라고 한 경우
        if self.target_vel == 0.8:
            output = 0.8  # self.pid_controller.pid(self.target_vel, self.current_vel)

        # 그냥 직진
        elif self.target_vel==1.5: #직선에서 살짝 틀어져서 감속
            output = 1.5
        else:
            output = 2.2

        self.control_angle.linear.x = abs(output)

        # self.control_angle.angular.z = steering_angle-0.001*self.angle_offset
        self.control_angle.angular.z = steering_angle + self.angle_offset
        # print(self.control_angle)
        self.vel_pub.publish(self.control_angle)
        # print(f'steering angle = {steering_angle}')

    # 조향각도를 -pi ~ +pi 범위로 정규화 (wrapping이라고 지칭)
    def normalize_angle(self, angle):
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle

    def stop(self):
        self.client.cancel_all_goals()
        twist = Twist()
        self.vel_pub.publish(twist)  # 속도를 0으로 설정하여 정지


def main():
    _ = NavigationClient()
    rospy.spin()


if __name__ == "__main__":
    main()

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

        self.pid_controller = pidControl()
        self.start_time = rospy.Time.now()

        self.curve_endpoint = 5 # 코너 끝까지 빠져나올 때까지 속도 유지하기 위함
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

        # print(f"왼쪽 포인트 좌표: {self.L_point[8]}")
        # print(f"오른쪽 포인트 좌표:{self.R_point[8]}")
        # print('R_curv', self.R_curv)

        L7_point = self.L_point[3].x
        R7_point = self.R_point[3].x
        # 1/25 19:46 ==> 4 였음
        reference_quat = self.goal_list[
            self.sequence + 5
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
                self.curve_endpoint = 6
                # 8km/h 일 때 twist2.0

            # if 200<=self.L_point<=240 and 480<=self.R_point<=510:
            #     #차선 유지 범위 내에 있음
            #     pass
            mid_point = (L7_point + R7_point) / 2

            if L7_point == 0 or R7_point == 0:  # 오른쪽이나 왼쪽 차선 없음
                if L7_point == 0:
                    if R7_point < 489:  # 너무 우측으로 치우침
                        self.angle_offset = 0.0000155 * (self.R_point[3].x - 498)

                    if R7_point < 494:  # 너무 우측으로 치우침
                        self.angle_offset = 0.0000145 * (self.R_point[3].x - 498)

                    elif 496 <= R7_point < 451:
                        pass
                    elif R7_point < 456:
                        self.angle_offset = 0.0000145 * (self.R_point[3].x - 498)
                    else:
                        self.angle_offset = 0.0000155 * (self.R_point[3].x - 498)

                elif R7_point == 0:
                    pass
                else:  # 둘다 0인 경우
                    self.angle_offset = 0
                self.angle_offset = 0
            else:
                if mid_point < 315:
                    self.angle_offset = 0.0000155 * (self.R_point[3].x - 322)
                if mid_point < 320:
                    self.angle_offset = 0.0000145 * (self.R_point[3].x - 322)
                elif mid_point < 325:
                    pass
                elif mid_point < 330:
                    self.angle_offset = 0.0000145 * (self.R_point[3].x - 322)
                else:
                    self.angle_offset = 0.0000155 * (self.R_point[3].x - 322)
            # if self.R_point[7].x>469:
            #     self.angle_offset=0.00009*(self.R_point[7].x-320)

            # elif self.L_point[7].x<251:
            #     self.angle_offset=-0.00009*(320-self.L_point[7].x)
            # else:qssssqs
            #     pass
            # print(f"차선 보정 값 : {self.angle_offset}")
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

            # self.lookahead_distance = (k_vel * v) + (k_curv * Curv)
            # self.lookahead_distance=0.25
            self.lookahead_distance = 0.79
        # 직진하는 경우
        else:
            v = 2
            k_vel = 0.085
            k_curv = 0.021

            # if Curv < 20:
            #     Curv = 30 #Ld 0.8 min)
            # elif Curv < 30:
            #     Curv = 31# Ld 0.821000
            # else:
            #     Curv = 32
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

        # 이 친구가 왔다리 갔다리
        if self.target_vel == 0.8:
            # print("I'm turning")
            # 1/25 19:46 ==> sequence + 4 일 때 0.973 이었음
            # 1/25 19:46 ==> sequence + 3 일 때 0.984 이었음

            # 1/25: test_test = 1. + test * 0.9843

            test_test = 1.0 + test * 0.99

            test_test = np.clip(test_test,5.7,10)

            print(f"곡선에서 gain값: {test_test}")
            self.corner_count+=1
            self.gain = test_test

            # if 0<=angle_diff_size<0.05:
            #     self.gain=1.03
            # elif 0.05<=angle_diff_size<0.08: #0.055대
            #     self.gain=1.069
            # elif 0.08<=angle_diff_size<0.10: #0.08대
            #     self.gain=1.071
            # elif 0.10<=angle_diff_size<0.15: #0.12, 0.18대
            #     self.gain=1.073
            # elif 0.15<=angle_diff_size<0.2:
            #     self.gain=1.078
            # elif 0.2<=angle_diff_size<0.25:
            #     self.gain=1.078
            # elif 0.25<=angle_diff_size<0.3:
            #     self.gain=1.080
            # elif 0.3<=angle_diff_size<0.35: #0.30대도 많이 나옴
            #     self.gain=1.082
            # elif 0.35<=angle_diff_size<0.4:
            #     self.gain=1.084
            # elif 0.4<=angle_diff_size<0.5:
            #     self.gain=1.086
            # else:
            #     self.gain=1.09 #이전 1.068

        else:

            # print("I'm straight")
            if test<0.08:
                test_test = 1.0 + test * 0.99
                test_test = np.clip(test_test, 1.0, 1.1)
                #print(f"똑바른 직선에서 gain값: {test_test}")
                self.corner_count=0
            else:
                if self.corner_count>4:
                    test_test = 1.0 + test * 0.99
                    test_test=np.clip(test_test, 5.7,58)
                    #print(f"코너 끝나고 수평 안맞을 때 gain값: {test_test}")
                else:
                    constant_multiplier = (5 - 1.5) / (2.9 - 0.8)
                    test_test = 1.0 + test * 2
                    test_test = (test_test - 1.5) / constant_multiplier
                    test_test = np.clip(test_test, 0.9, 5.7)
                    self.target_vel=1.8


                    
                    print(f"직선에서 어긋났을때 gain값: {test_test}")
                    


            self.gain = test_test

            # if test < 0.35:
            #     print('abs(angle_difference) < 0.35!, angle_difference :', abs(angle_difference))
            #     gain = 1.031

            # elif test < 0.5:
            #     print('abs(angle_difference) < 0.5!, angle_difference :', abs(angle_difference))
            #     gain = 1.0468

            # else:
            #     print('abs(angle_difference) > 0.5!, angle_difference :', abs(angle_difference))
            #     gain = 1.067

        # 살짝 잘 됐던 코드
        # if self.target_vel==0.8:
        #     if angle_difference<0.2:
        #         gain=1
        #     elif 0.2<=angle_difference<0.5:
        #         gain=1.04
        #     else:
        #         gain=1.068대

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
        
        print(steering_angle)
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

    def LKAS(self, L_index, R_index, Curve):
        # 직선인가 커브인가
        straight = 5
        if straight < Curve:
            # 일단 차선이 둘다 있는가 없는가
            ratio = abs(L_index - R_index)
            R0 = 10  # 기준 크기

        if L_index == None and R_index != None:
            L_index = R0
        elif R_index == None and L_index != None:
            R_index = R0
        elif R_index == None and L_index == None:  # 차선이 양쪽 다 없는 경우는 pass
            L_index = R_index = R0
        # 좌회전 시 왼쪽 차선의 곡률만 고려 > 최대한 understeer를 내면서 붙는 방향으로 설정
        # 1. oversteer가 발생하는 경우는 핸들을 좀 풀어준다
        # 2. understeer가 발생하는 경우는 핸들을 더 준다
        return

    def stop(self):
        self.client.cancel_all_goals()
        twist = Twist()
        self.vel_pub.publish(twist)  # 속도를 0으로 설정하여 정지


def main():
    _ = NavigationClient()
    rospy.spin()


if __name__ == "__main__":
    main()

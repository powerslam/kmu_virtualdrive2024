#! /usr/bin/env python3

import tf
import rospy
import pickle
import actionlib
import numpy as np

from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

from lane_detection.msg import LaneInformation, PixelCoord
from obstacle_detect.msg import ObstacleInfo, ObstacleInfoArray

from actionlib_msgs.msg import GoalStatus

import matplotlib.pyplot as plt


class Total:
    def __init__(self):
        # node 정보 초기화
        rospy.init_node("navigation_client")

        # 현재 수행해야 할 미션이 무엇인지 가리키는 함수
        self.MISSION = 0

        self.TEB_Planner = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.TEB_Planner.wait_for_server()

        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

        self.goal_list = []
        self.sequence = 0  # 받아온 좌표모음의 index

        self.obstacle_point = 0

        with open("/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl", "rb") as file:
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

        # SLAM 구간용 코드
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.slam_goal = MoveBaseGoal()
        self.slam_goal.target_pose.header.frame_id = 'map'
        self.slam_goal.target_pose.pose.position.x = 18.075382185152254
        self.slam_goal.target_pose.pose.position.y = -9.681479392662882
        self.slam_goal.target_pose.pose.orientation.z = 0
        self.slam_goal.target_pose.pose.orientation.w = 1

        self.vehicle_length = 0.26

        # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =0.8 일때( 3km/h) 0.35
        self.lookahead_distance = 0.8

        self.angle_offset = 0
        self.gain = 1.0

        self.is_current_vel = False
        self.is_state = False
        self.target_vel = 0
        self.current_vel = 0.0

        self.NO_RIGHTLINE = False
        self.NO_LEFTLINE = False 

        self.start_time = rospy.Time.now()

        # 20이 기본 세팅 값
        self.CURVE_ENDPOINT = 20
        self.curve_endpoint = self.CURVE_ENDPOINT # 코너 끝까지 빠져나올 때까지 속도 유지하기 위함
        
        # self.curve_endpoint = True
        self.curve_start = True
        self.corner_count = 0
        self.control_angle = Twist()

        self.stop_flag = False

        # AMCL pose Subscribe
        self.now_pose = None
        self.now_orientation = None
        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        # 차선 정보 가져오기
        self.L_curv, self.R_curv, self.L_point, self.R_point = 0, 0, [], []
        rospy.Subscriber("/lane_information", LaneInformation, self.lane_callback)
        
        # 현재 차량의 속도 받아오기
        rospy.Subscriber("/odometry/filtered", Odometry, self.ld_callback)

        self.obstacles = None
        rospy.Subscriber('/obstacle_information', ObstacleInfoArray, self.mission2)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

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

    def ld_callback(self, data: Odometry):
        # print(self.MISSION)
        # R_curv 가 0인 경우 대책을 세워야 함
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

        elif self.obstacle_point > 0:
            self.lookahead_distance = 0.3

        else: # 직진하는 경우
            v = 2
            k_vel = 0.085
            k_curv = 0.021
            Curv = 34

            self.lookahead_distance = (k_vel * v) + (k_curv * Curv)
        # print(f'Ld값은 :{self.lookahead_distance}')

        if self.MISSION == 0:
            self.mission1()
        
        elif self.MISSION == 1:
            # print('mission1')
            self.run()

        elif self.MISSION == 2:
            # print('mission2')
            self.run()

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )

        # tf.transformations.quaternion_from_euler
 
        return euler[2]  # yaw

    def mission1(self): # MISSION1 슬램 구간 탈출
        if not self.now_pose: 
            return

        if self.client.get_state() == GoalStatus.SUCCEEDED:
            # self.MISSION = -1
            self.MISSION += 1

        elif self.client.get_state() != GoalStatus.ACTIVE:
            self.client.send_goal(self.slam_goal)

    def mission2(self, msg: ObstacleInfoArray): # 장애물 구간
        if self.obstacle_point != 0: return
        if self.MISSION != 1: return
        if not self.now_pose: return
        
        obstacle_infos = msg.obstacles
        if len(obstacle_infos):
            # self.stop()
            # obstacle_infos[0].distance < 2.4 # 이게 적절하게 멈추는 듯
            if obstacle_infos[0].distance < 2.4:
                # self.stop_flag = True
                # lidar 좌표계 => 차량 좌표계
                obst_x = obstacle_infos[0].y + self.now_pose.x + 0.11
                obst_y = -obstacle_infos[0].x + self.now_pose.y

                target_x = obst_x
                target_y = obst_y + 0.3
                print('now_pose', self.now_pose)
                print('target', target_x, target_y)

                mid_point = (target_y + self.now_pose.y) / 2.

                self.obstacle_point  = int(obstacle_infos[0].distance / 0.2)

                # 그래프 상에서 우리는 오른쪽 아래에서 차량이 굴러가기 때문
                new_y = np.linspace(-0.9640275800758169, 0.9640275800758169, self.obstacle_point)[::-1]
                new_x = -np.arctanh(new_y) * (obstacle_infos[0].distance / 4)

                way_y = new_y + new_y[0] + self.now_pose.y # way point 용 y 좌표
                way_x = new_x - new_x[0] + self.now_pose.x 

                # plt.scatter(way_y, way_x)
                # plt.show()

                for seq in range(self.sequence, self.sequence + self.obstacle_point):
                    idx = seq - self.sequence

                    self.goal_list[seq].target_pose.pose.position.x = way_x[idx]
                    self.goal_list[seq].target_pose.pose.position.y = way_y[idx]

                    _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, -(1 / (1 - new_y[idx] ** 2)))
    
                    self.goal_list[seq].target_pose.pose.orientation.x = 0.
                    self.goal_list[seq].target_pose.pose.orientation.y = 0.
                    self.goal_list[seq].target_pose.pose.orientation.z = qz
                    self.goal_list[seq].target_pose.pose.orientation.w = qw

                print('make_path')
                print(self.goal_list[self.sequence:self.sequence+self.obstacle_point])

                print('\n\nafter_path')
                print(self.goal_list[self.sequence+self.obstacle_point:self.sequence+self.obstacle_point+10])

        if self.stop_flag: 
            # print(self.goal_list[self.sequence:self.sequence+self.obstacle_point])
            self.stop()

    def run(self):
        if self.stop_flag: return
        if not self.now_pose: return
        if (
            self.dist(self.goal_list[self.sequence].target_pose.pose.position)
            < self.lookahead_distance
        ):
            if self.sequence >= len(self.goal_list): return
            self.sequence += 1

            if self.obstacle_point > 0:
                self.obstacle_point -= 1

        dy = self.goal_list[self.sequence].target_pose.pose.position.y - self.now_pose.y
        dx = self.goal_list[self.sequence].target_pose.pose.position.x - self.now_pose.x

        # Pure Pursuit 알고리즘 적용
        angle_to_target = np.arctan2(dy, dx)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        angle_difference = angle_to_target - yaw
        angle_difference = self.normalize_angle(angle_difference)
        
        test = abs(angle_difference)

        corner_gain_min = 4.93

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
                        print('차선인식 못함 ')
                        pass

                    else:
                        constant_multiplier = (5 - 1.5) / (2.9 - 0.8)
                        test_test = 1.0 + test * 2
                        test_test = (test_test - 1.5) / constant_multiplier
                        test_test = np.clip(test_test, 5.0, 5.7)
                        self.target_vel = 2.0

                        print(f"직선에서 어긋났을때 gain값: {test_test}")
                    
            self.gain = test_test

        steering_angle = self.gain * np.arctan2(2.0 * self.vehicle_length * np.sin(angle_difference) / self.lookahead_distance, 1.0)
        self.control_angle = Twist()

        if self.target_vel == 0.8:
            output = 0.8

        elif self.obstacle_point > 0:
            output=0.6

        elif self.target_vel == 1.5:
            output = 1.5

        else:
            output = 2.2

        self.control_angle.linear.x = abs(output)
        self.control_angle.angular.z = steering_angle + self.angle_offset
        self.vel_pub.publish(self.control_angle)

    def normalize_angle(self, angle):
        # 조향각도를 -pi ~ +pi 범위로 정규화 (wrapping이라고 지칭)
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

    def stop(self):
        print('stop?')
        self.client.cancel_all_goals()

        twist = Twist()
        self.vel_pub.publish(twist)  # 속도를 0으로 설정하여 정지

if __name__ == "__main__":
    nc = Total()
    rospy.spin()

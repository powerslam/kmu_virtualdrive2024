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

from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

from lane_detection.msg import LaneInformation, PixelCoord

# 신호등 토픽
from morai_msgs.msg import GetTrafficLightStatus

# 카메라
from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Float64

# 카메라 정보
from obstacle_detect.msg import ObstacleInfo, ObstacleInfoArray

# 라이다 정보
from obstacle_detect.msg import LidarObstacleInfo, LidarObstacleInfoArray, RotaryArray
from actionlib_msgs.msg import GoalStatus

class Total:
    def __init__(self):
        # node 정보 초기화
        rospy.init_node("navigation_client")

        # 현재 수행해야 할 미션이 무엇인지 가리키는 함수
        self.MISSION = 0

        self.TEB_Planner = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.TEB_Planner.wait_for_server()

        self.vel_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        
        self.stop_data = Float64(data=0)

        self.sequence = 0  # 받아온 좌표모음의 index
        self.obstacle_point = 0

        # SLAM 구간용 코드(mission 1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.slam_goal = MoveBaseGoal()
        self.slam_goal.target_pose.header.frame_id = 'map'
        self.slam_goal.target_pose.pose.position.x = 18.075382185152254
        self.slam_goal.target_pose.pose.position.y = -9.681479392662882
        self.slam_goal.target_pose.pose.orientation.z = 0
        self.slam_goal.target_pose.pose.orientation.w = 1


        # 장애물 구간 (SLAM 구간 종료 지점 ~ 로터리 앞 정지선)
        self.mission23_goal = []
        with open("/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/mission23.pkl", "rb") as file:
            pt_list = pickle.load(file)

            for _pt in pt_list:
                pt = MoveBaseGoal()
                pt.target_pose.header.frame_id = "map"
                pt.target_pose.pose.position.x = _pt.position.x
                pt.target_pose.pose.position.y = _pt.position.y
                pt.target_pose.pose.orientation.z = _pt.orientation.z
                pt.target_pose.pose.orientation.w = _pt.orientation.w

                self.mission23_goal.append(pt)

        # 로터리 및 좌회전 미션(로터리 앞 정지선 ~ 골 지점까지)
        self.mission45_goal = []
        with open("/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/mission45.pkl", "rb") as file:
            pt_list = pickle.load(file)

            for _pt in pt_list:
                pt = MoveBaseGoal()
                pt.target_pose.header.frame_id = "map"
                pt.target_pose.pose.position.x = _pt.position.x
                pt.target_pose.pose.position.y = _pt.position.y
                pt.target_pose.pose.orientation.z = _pt.orientation.z
                pt.target_pose.pose.orientation.w = _pt.orientation.w

                self.mission45_goal.append(pt)

        self.goal_list = [self.mission23_goal, self.mission45_goal]

        self.vehicle_length = 0.26

        # 저속3니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =0.8 일때( 3km/h) 0.35
        self.lookahead_distance = 0.8

        self.angle_offset = 0
        self.gain = 1.0

        self.is_current_vel = False
        self.is_state = False
        self.target_vel = 0
        self.current_vel = 0.0

        self.NO_RIGHTLINE = False
        self.NO_LEFTLINE = False 

        self.NO_REF = False 

        # 20이 기본 세팅 값
        self.CURVE_ENDPOINT = 10   #@손희문 : 이전 20 에서 5으로 수정 > 10으로 늘릴까유>?
        self.curve_endpoint = self.CURVE_ENDPOINT # 코너 끝까지 빠져나올 때까지 속도 유지하기 위함
        
        # self.curve_endpoint = True
        self.curve_start = True
        self.corner_count = 0
        self.control_angle = Twist()

        self.stop_flag = False
        
        # 장애물은 2단계로 이루어져 있음. 정적/동적 or 동적/정적
        self.obstacle_stage = 1

        # 그런데, 처음이 정적인지, 동적인지 모르니까 이걸로 처음이 뭐였는지 체크함
        self.obstacle_type = None

        self.prev_obst_point = None
        self.yaw_diff_mission1 = 0

        self.rotary_exit_flag = False

        self.TURN_VEL = 1200
        self.INPO_VEL = 1800
        self.STRA_VEL = 2400

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

        # # 라이다 장애물 받아오기
        self.obstacles = None
        rospy.Subscriber('/lidar_obstacle_information', LidarObstacleInfoArray, self.mission1)

        # 로터리 변수, 플래그
        self.move_car_ori = None
        self.move_car_dis = None
        self.car_flag = False
        rospy.Subscriber('/rotary_info', RotaryArray, self.mission2)

        rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.mission3)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def lane_callback(self, msg: LaneInformation):
        if self.rotary_exit_flag:
            self.stop()
            print("i'm off")
            return

        # 직선 = 곡률 반지름이 엄청 커지기에 곡률은 작다straight 보다 작으면
        # 곡선
        self.L_curv = msg.left_gradient
        self.R_curv = msg.right_gradient
        self.L_point = msg.left_lane_points
        self.R_point = msg.right_lane_points

        L3_point = self.L_point[3].x
        R3_point = self.R_point[3].x

        l_pt_obstacle_point = len([1 for pt in self.L_point if pt.x == 0])
        r_pt_zero_obstacle_point = len([1 for pt in self.R_point if pt.x == 0])

        # 1/25 19:46 ==> 4 였음
        reference_quat = self.goal_list[int(self.MISSION > 1)][
            min(self.sequence + 6, len(self.goal_list[int(self.MISSION > 1)]) - 1)
        ].target_pose.pose.orientation  # 이전 4
        reference_yaw = self.get_yaw_from_orientation(reference_quat)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        self.yaw_diff_mission1 = abs(yaw - reference_yaw)
        # @차선 변경 조건 각도 (30도? 40도 추가해야함)
        # # # # print('yaw', yaw - reference_yaw)

        mid_point = (L3_point + R3_point) / 2

        mid_r_point = 498
        mid_l_point = 146

        C_straight = 1.8

        if self.obstacle_point > 0:
            self.target_vel = self.TURN_VEL

        elif abs(yaw - reference_yaw) > 0.3:  # 각도 차이가 어느정도 난다. 회전해야함
            self.target_vel = self.TURN_VEL
            mid_point = (L3_point + R3_point) / 2

            mid_r_point = 498
            mid_l_point = 146

            C_straight = 1.8
            # print('r_pt_zero_obstacle_point', r_pt_zero_obstacle_point)
            if 4 < r_pt_zero_obstacle_point : self.NO_REF = True
            
            if self.curve_start:
                # # print("커브구간 진입  ")
                self.curve_start = False
            
            if yaw - reference_yaw < 0: #좌회전임
                # # print("좌회전 중 ")
                if L3_point == 0 and R3_point!=0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                    self.NO_LEFTLINE=True
                    self.NO_RIGHTLINE=False
                    # print("좌회전 중 - 왼쪽 차선 X  ")
                
                elif R3_point == 0 and L3_point != 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                    self.angle_offset = -C_straight * (L3_point - mid_l_point) / 200
                    self.NO_RIGHTLINE=True 
                    self.NO_LEFTLINE=False  
                    # print("좌회전 중 - 오른쪽 차선 X ")

                elif R3_point == 0 and L3_point==0:
                    self.angle_offset=0
                    self.NO_RIGHTLINE=True 
                    self.NO_LEFTLINE=True 
                    # print("양쪽 다 인식 x ")
                    self.R_curv = 480

                else:
                    if not self.NO_REF:
                        self.angle_offset = C_straight * (322 - mid_point) / 500
                        self.NO_RIGHTLINE = self.NO_LEFTLINE=False 
                        # print("좌회전 중 - 양쪽 차선 다 인식")
                    
                    else:
                        self.angle_offset=0
                
                if self.R_curv == 480:
                    self.angle_offset = 0.35
                    # print("R_curve 480  !!")

            else:  #우회전임 
                # print("우회전 중 ", yaw - reference_yaw, 'yaw', yaw, 'ref_yaw', reference_yaw)
                if L3_point == 0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                    self.angle_offset = C_straight * (mid_r_point - R3_point) / 500
                    self.NO_LEFTLINE=True
                    self.NO_RIGHTLINE=False

                elif R3_point == 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                    self.NO_RIGHTLINE=True 
                    self.NO_LEFTLINE=False  
                
                else:
                    self.angle_offset = C_straight * (322 - mid_point) / 500
                    self.NO_RIGHTLINE=self.NO_LEFTLINE=False 


            # 속도 줄이기
        else:
            self.NO_REF = False
            if (
                self.lookahead_distance==0.79 and self.curve_endpoint!=0 and not self.curve_start 
            ):  # Ld값 짧은상태=코너 주행중이었다면, 2번 속도 증가무시
                # print("좌회전이나 우회전의 꼬리부분 ")
                self.target_vel = self.TURN_VEL
                # self.curve_endpoint = False회
                self.curve_endpoint -= 1

                if yaw-reference_yaw < 0: #좌회전임
                    # # print("좌회전 중 ")
                    if L3_point == 0 and R3_point!=0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                        self.NO_LEFTLINE=True
                        self.NO_RIGHTLINE=False
                        # print("좌회전 중 - 왼쪽 차선 X  ")
                    
                    elif R3_point == 0 and L3_point != 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                        self.angle_offset = -C_straight * (L3_point - mid_l_point) / 200
                        self.NO_RIGHTLINE=True 
                        self.NO_LEFTLINE=False  
                        # print("좌회전 중 - 오른쪽 차선 X ")

                    elif R3_point == 0 and L3_point==0:
                        self.angle_offset=0
                        self.NO_RIGHTLINE=True 
                        self.NO_LEFTLINE=True 
                        # print("양쪽 다 인식 x ")
                        self.R_curv = 480

                    else:
                        if not self.NO_REF:
                            self.angle_offset = C_straight * (322 - mid_point) / 500
                            self.NO_RIGHTLINE = self.NO_LEFTLINE=False 
                            # print("좌회전 중 - 양쪽 차선 다 인식")
                        
                        else:
                            self.angle_offset=0
                    
                    if self.R_curv == 480:
                        self.angle_offset = 0.35
                        # print("R_curve 480  !!")

                else:  #우회전임 
                    # print("우회전 중 ")
                    if L3_point == 0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                        self.angle_offset = C_straight * (mid_r_point - R3_point) / 500
                        self.NO_LEFTLINE=True
                        self.NO_RIGHTLINE=False
                    elif R3_point == 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                        self.NO_RIGHTLINE=True 
                        self.NO_LEFTLINE=False  
                    else:
                        self.angle_offset = C_straight * (322 - mid_point) / 500
                        self.NO_RIGHTLINE=self.NO_LEFTLINE=False 


            else:
                # print("고속 직진 중 ")
                self.control_angle.linear.x = 2.2
                self.target_vel = self.STRA_VEL
                # self.curve_endpoint = True
                self.curve_endpoint = self.CURVE_ENDPOINT
                # 8km/h 일 때 twist 2.0
                self.curve_start=True

                mid_point = (L3_point + R3_point) / 2

                mid_r_point = 498
                mid_l_point = 146

                if L3_point == 0 or R3_point == 0:  # 오른쪽이나 왼쪽 차선 없음
                    if L3_point == 0: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                        self.angle_offset = C_straight * (mid_r_point - R3_point) / 1000
                        self.NO_LEFTLINE=True
                        self.NO_RIGHTLINE=False
                    elif R3_point == 0: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                        self.angle_offset = -C_straight * (L3_point - mid_l_point) / 1000
                        self.NO_RIGHTLINE=True
                        self.NO_LEFTLINE=False  

                    else:  # 둘다 0인 경우 > 아무래도 패쓰
                        self.angle_offset = 0
                        self.NO_LEFTLINE=self.NO_RIGHTLINE=True

                else:
                    self.angle_offset = C_straight * (322 - mid_point) / 10000
                    #self.angle_offset=0
                    self.NO_RIGHTLINE=self.NO_LEFTLINE=False

            """
            self.angle_offset
            회전을 해야될 때 일단 차선 보지 말자 
            직진주행일때, 양쪽의 선을 보자 
            """
        self.angle_offset = 0

    def ld_callback(self, data: Odometry):
        # # # # print(self.MISSION)
        # R_curv 가 0인 경우 대책을 세워야 함
        v = data.twist.twist.linear.x
        # print('속도', v)

        # 오른쪽 차선의 기울기
        Curv = abs(self.R_curv)

        # 만약 로터리 진입했다면 장애물 자랑 속도 맞추기(로터리 끝나느 waypoint까지)
        if self.car_flag:
            v = 0.3

        # 회전하는 경우
        if v < 1.:  # 맨 처음 출발할때
            #self.lookahead_distance = 0.8840000000000001
            # print('turning')
            self.lookahead_distance = 0.6

        # elif v < 0.9:
        #     #self.lookahead_distance = 0.79
        #     self.lookahead_distance = 0.8
        
        else: # 직진하는 경우
            #self.lookahead_distance = 0.8840000000000001
            self.lookahead_distance = 1.
        
        if self.MISSION == 0:
            # print('mission0 hi?')
            self.mission0()
        
        else:
            self.run()

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )

        return euler[2]  # yaw

    def mission0(self): # MISSION0 슬램 구간 탈출
        if not self.now_pose:
            return
        
        # print('self.now_pose live')

        if self.client.get_state() == GoalStatus.SUCCEEDED:
            self.client.cancel_goal()
            # TEB 종료
            os.system('rosnode kill /throttle_interpolator')
            os.system('rosnode kill /move_base')
            self.MISSION = 1

        elif self.client.get_state() != GoalStatus.ACTIVE:
            self.client.send_goal(self.slam_goal)

    def mission1(self, msg: LidarObstacleInfoArray): # 장애물 구간
        # rotary 때문에 임시로 멈추게 함
        if self.MISSION != 1: return

        if not self.now_pose: return

        if self.obstacle_type and self.obstacle_stage != 2 and self.target_vel == self.TURN_VEL and self.R_curv < 0.1:
            print('turn')
            self.obstacle_stage = 2
            return

        obstacle_infos = msg.obstacle_infos
        if not len(obstacle_infos): return

        # mission 2 인 경우
        if self.obstacle_stage == 1:
            for info in obstacle_infos:
                print('장애물 발견')
                dist = max(0., np.hypot(info.obst_x, info.obst_y) - 0.21)
                # # print('dist', dist)

                # 장애물 종류가 결정된 경우
                if self.obstacle_type:
                    print("엥?")
                    if self.obstacle_type == 'd':
                        if info.obst_x > 0:
                            self.stop_flag = False

                        elif info.obst_x > -1.38 and info.obst_y > 0.1 and dist < 1.35:
                            self.stop_flag = True
                            self.stop()

                    else: # 정적 장애물인 경우
                        # 이 경우 장애물 회피 경로가 모두 완료된 경우 뭔가를 처리해야 함
                        pass
                
                else:
                    print('장애물 미결정', info)

                    # 근데 이러지 말고 그냥 info.obst_x 가 일정 범위 밖인 경우 동적/정적으로 갈라도 될 듯
                    # 순수하게 앞에 장애물이 있는 경우 ==> 정적 장애물인 경우
                    if -0.2 < info.obst_x < 0.2: #@손희문 2.4로 하면 벽면을 봐버리는 문제 
                        print("정적 장애물 인식")
                        self.obstacle_type = 's'
                        self.dy_flag = False # False는 현재 자동차가 2차선, True는 현재 자동차가 1차선

                        # 임시
                        dy_ob_x = info.obst_y + self.now_pose.x # amcl상에서 장애물의 x좌표
                        dy_ob_y = -info.obst_x + self.now_pose.y # amcl상에서 장애물의 y좌표
                        self.stop_flag  = True
                        self.stop()
                        # 차선 변경 실시
                        
                        # 장애물 옆으로 회피
                        if self.dy_flag == False: # 2차선일 때
                            #1차선으로 이동
                            self.dy_flag = True
                            target_x = dy_ob_x        # amcl상에서 이동해야 할 x좌표
                            target_y = dy_ob_y + 0.35 # amcl상에서 이동해야 할 y좌표
                            self.stop_flag  = False
                            self.create_trajectory( target_x, target_y, dist)
                            

                            #y축을 빼야 하는 경우도 있는걸 염두

                        elif self.dy_flag == True: # 1차선일 때
                            #2차선으로 이동
                            self.dy_flag = False
                            target_x = dy_ob_x        # amcl상에서 이동해야 할 x좌표
                            target_y = dy_ob_y - 0.35 # amcl상에서 이동해야 할 y좌표
                            self.stop_flag  = False
                            self.create_trajectory( target_x, target_y, dist)
                            

                            #y축을 빼야 하는 경우도 있는걸 염두









                        pass
                    
                    # 동적 장애물인 경우
                    else:
                        if self.obstacle_type is None:
                            self.obstacle_type = 'd'
                            return
                    
            # for문에 달린 else
            else:
                if not obstacle_infos:
                    self.stop_flag = False
                    return

            # # # print('self.obstacle_type', self.obstacle_type)

        elif self.obstacle_stage == 2:
            # 위랑 다르게 d 인경우 정적, s인 경우 동적으로 구현하면 됨
            # print('next stage!')
            # self.stop_flag = True
            # self.stop()
            return
        
        # 두 단계 모두 지나면 장애물 체크할 필요가 없기 때문
        else: return

        if self.stop_flag: 
            # # # # print(self.goal_list[self.sequence:self.sequence+self.obstacle_point])
            self.stop()

    def mission2(self, msg: RotaryArray):
        self.move_car = msg.moving_cars
        self.min_dis = min(self.move_car,key = lambda x: x.dis)

    def mission3(self, msg: GetTrafficLightStatus):
        if self.rotary_exit_flag:
            self.stop()
            self.rotary_exit_flag = False
            print('hi?')
            # self.MISSION += 1
        
        # 정지선이 나올 때까지 일반 주행
        # 신호에 따라서 정지할 지 말지 정하기
        # 이때, 동적 장애물, 정적 장애물 등 장애물 회피도 포함되어야 함

    def dy_obstacle(self, msg: LidarObstacleInfoArray):
        dy_obstacle_infos = msg.obstacle_infos
        self.dy_flag = False # False는 현재 자동차가 2차선, True는 현재 자동차가 1차선

        if not len(dy_obstacle_infos): return

        for dy_info in dy_obstacle_infos:
                dy_dist = np.sqrt((dy_info.obst_x**2)+(dy_info.obst_y**2))

                if -0.1 < dy_info.obst_x < 0.1 and dy_info.obst_y < 2.0 and dy_dist == 1.5: # 정면에 장애물이 있을 때
                    self.stop()

                    dy_ob_x = dy_info.obst_x + self.now_pose.x # amcl상에서 장애물의 x좌표
                    dy_ob_y = dy_info.obst_y + self.now_pose.y # amcl상에서 장애물의 y좌표

                    
                    self.create_trajectory( dy_ob_x, dy_ob_y, dy_dist)
                    # 장애물 옆으로 회피
                    if self.dy_flag == False: # 2차선일 때
                        #1차선으로 이동
                        self.dy_flag = True
                        target_x = dy_ob_x - 0.35 # amcl상에서 이동해야 할 x좌표
                        target_y = dy_ob_y        # amcl상에서 이동해야 할 y좌표

                        #y축을 빼야 하는 경우도 있는걸 염두

                    elif self.dy_flag == True: # 1차선일 때
                        #2차선으로 이동
                        self.dy_flag = False
                        target_x = dy_ob_x + 0.35 # amcl상에서 이동해야 할 x좌표
                        target_y = dy_ob_y        # amcl상에서 이동해야 할 y좌표

                        #y축을 빼야 하는 경우도 있는걸 염두

                elif dy_info.obst_x < -0.1 or dy_info.obst_x > 0.1:
                    #직진
                    pass

        # 더 이상 장애물이 보이지 않으니까 원래 waypoint로 이동
    
    def create_trajectory(self, tgt_x, tgt_y, dist):
        x = np.array([self.now_pose.x, self.now_pose.x + dist * 0.3, tgt_x - dist * 0.3, tgt_x])
        y = np.array([self.now_pose.y, self.now_pose.y       , tgt_y       , tgt_y])

        print('trajectory x', x)
        print('trajectory y', y)

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

        # print(orientation_new)

        self.obstacle_point = 0
        for seq in range(self.sequence + 1, self.sequence + num):
            self.obstacle_point += 1
            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.position.x = x_new[self.obstacle_point]
            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.position.y = y_new[self.obstacle_point]
            _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, orientation_new[self.obstacle_point])

            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.orientation.x = 0.
            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.orientation.y = 0.
            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.orientation.z = qz
            self.goal_list[int(self.MISSION > 1)][seq].target_pose.pose.orientation.w = qw
        
        print("회피경로 생성 완료 ")

        plt.figure(figsize = (dist, 0.5))
        plt.plot(x_new, y_new, 'b')
        plt.plot(x_scale, y_scale, 'ro')
        plt.plot(x_new, orientation_new, 'g')
        
        plt.title('Cubic Hermite Spline Interpolation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def run(self):
        if not self.now_pose: return

        if not self.rotary_exit_flag and self.dist(self.goal_list[int(self.MISSION > 1)][self.sequence].target_pose.pose.position) < self.lookahead_distance:
            if self.obstacle_point > 0:
                self.obstacle_point -= 1
            
            # print('seq', self.sequence)
            if self.sequence == len(self.goal_list[int(self.MISSION > 1)]) - 1:
                self.stop_flag = True
                self.stop()
                # print("finish")
                
                if 0 < self.min_dis.dis < 1.5:
                    self.stop_flag = True 
                    self.stop()
                    
                else:
                    # print('enter')
                    self.target_vel = self.TURN_VEL # 이거 다음은 회전이니까
                    self.rotary_exit_flag = True
                    self.sequence = 0
                    self.stop_flag = False
                    self.MISSION += 1
                    return

            else :
                self.sequence += 1
            

        # 차량 좌표계는 (차량 x축) = (일반 y축), (차량 y축) = -(일반 x축)
        dy = self.goal_list[int(self.MISSION > 1)][self.sequence].target_pose.pose.position.x - self.now_pose.x
        dx = self.now_pose.y - self.goal_list[int(self.MISSION > 1)][self.sequence].target_pose.pose.position.y 

        # print('dy', dy, 'dx', dx)

        # Pure Pursuit 알고리즘 적용
        # 목표지점과 현재 지점 간 각도
        angle_to_target = np.arctan2(dx, dy)
        # print('angel_to_target', angle_to_target)
        
        # 현재 차량의 pose
        yaw = -self.get_yaw_from_orientation(self.now_orientation)
        # print('yaw', yaw)
        
        # 목표 지점과 나의 pose의 차이를 계산
        angle_difference = angle_to_target - yaw
        
        # 이거는 각도 정규화
        # angle_difference = self.normalize_angle(angle_difference)
        # print('angle_diff', angle_difference)

        # 
        test = abs(angle_difference)

        # 원래 4.93 , 1.8 중 뭐지?
        corner_gain_min = 1.53     

        if self.target_vel == self.TURN_VEL:
            test_test = 1.0 + test * 1.15  #0.99

            test_test = np.clip(test_test, corner_gain_min, 2.4)
            if self.R_curv==480 or self.L_curv==480:
                test_test=2.6
            ## print(f"곡선에서 gain값: {test_test}")
            self.corner_count += 1
            self.gain = test_test
            
        else:
            if test < 0.04:
                test_test = 1.0 + test * 1.4
                test_test = np.clip(test_test, 1.0, 2.0)
                ## print(f"똑바른 직선에서 gain값: {test_test}")
                self.corner_count = 0
            
            else:
                if self.corner_count > 4:
                    test_test = 1.0 + test * 1.15
                    test_test = np.clip(test_test, corner_gain_min, 2.4)
                    ## print(f"코너 끝나고 수평 안맞을 때 gain값: {test_test}")

                else:
                    if self.NO_LEFTLINE or self.NO_RIGHTLINE: #둘 중에 하나라도 차선인식이 안되는 경우 
                        ## print('차선인식 못함 ')
                        test_test=1
                        pass

                    else:
                        constant_multiplier = (5 - 1.5) / (2.9 - 0.8)
                        test_test = 1.0 + test * 2
                        test_test = (test_test - 1.5) / constant_multiplier
                        test_test = np.clip(test_test, 1.4 , 1.5)
                        if not self.rotary_exit_flag:
                            self.target_vel = self.INPO_VEL

                        ## print(f"직선에서 어긋났을때 = tgt vel 2.0일때 gain값: {test_test}")
                    
            self.gain = test_test

        self.gain = 1
        steering_angle = self.gain * np.arctan2(2.0 * self.vehicle_length * np.sin(angle_difference) / self.lookahead_distance, 1.0)
        
        # @TODO: 이 친구가 값을 적절하게 mapping 시켜주는 것은 아님 
        #        ==> 따라서 최대 steering_angle 값을 보고 mapping 시켜줄 수 있는 값을 찾는게 중요

        # 2/1 은 pure pursuit 조정하는 1팀(2인)
        #        각종 미션에 대해 만족하는 코드를 짜는 1팀(2인)
        steering_angle = np.clip(steering_angle, -0.5, 0.5)
        
        steer = steering_angle + 0.5
        # print(steer, steering_angle)

        # if self.target_vel <= self.TURN_VEL:
        #     output = self.target_vel
        
        # elif self.target_vel == self.INPO: 
        #     output = 1.5
        
        # else:
        #     output = 2.2

        # self.control_angle.linear.x = abs(output)
        
        # self.control_angle.angular.z = steering_angle + self.angle_offset
        
        if not self.stop_flag:
            # print(self.target_vel, steering_angle)
            self.vel_pub.publish(Float64(data = self.target_vel))
            self.steer_pub.publish(Float64(data = steer))

    def normalize_angle(self, angle):
        # 조향각도를 -pi ~ +pi 범위로 정규화 (wrapping이라고 지칭)
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

    def stop(self):
        self.vel_pub.publish(Float64(data=0))  # 속도를 0으로 설정하여 정지

if __name__ == "__main__":



    nc = Total()
    rospy.spin()
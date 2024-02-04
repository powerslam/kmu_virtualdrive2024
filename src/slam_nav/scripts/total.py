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
        # node 정보 초기화
        rospy.init_node("navigation_client")

        # 현재 수행해야 할 미션이 무엇인지 가리키는 함수
        self.MISSION = 0

        self.TEB_Planner = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.TEB_Planner.wait_for_server()

        self.vel_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        self.path_pub = rospy.Publisher('/global_pah', Path, queue_size=1)
        
        self.stop_data = Float64(data=0)

        self.sequence = 0  # 받아온 좌표모음의 index
        self.obstacle_point = 0

        # SLAM 구간용 코드(mission 1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.slam_goal = MoveBaseGoal()
        self.slam_goal.target_pose.header.frame_id = 'map'
        self.slam_goal.target_pose.pose.position.x = 16.39718636233998
        self.slam_goal.target_pose.pose.position.y = -9.907622352044239
        self.slam_goal.target_pose.pose.orientation.z = 0
        self.slam_goal.target_pose.pose.orientation.w = 1

        self.global_path = Path()
        self.global_path.header.frame_id = 'map'

        self.position_offset_x = 0
        self.position_offset_y = 0

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

                pt = PoseStamped()
                pt.header.frame_id = 'map'
                pt.pose.position.x = _pt.position.x
                pt.pose.position.y = _pt.position.y
                pt.pose.orientation.z = _pt.orientation.z
                pt.pose.orientation.w = _pt.orientation.w
                
                self.global_path.poses.append(pt)

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

                pt = PoseStamped()
                pt.header.frame_id = 'map'
                pt.pose.position.x = _pt.position.x
                pt.pose.position.y = _pt.position.y
                pt.pose.orientation.z = _pt.orientation.z
                pt.pose.orientation.w = _pt.orientation.w
                
                self.global_path.poses.append(pt)

        '''
        0.5029473684905209, 0.49642105277700965, 0.49184210509822124, 0.49184210509822124, 0.4857894736398328, 
                             0.4857894736398328, 0.481263157340185, 0.481263157340185, 0.481263157340185, 0.47521052611790693, 
                             0.47521052611790693, 0.4727368419618505, 0.4727368419618505, 0.4647894736398328, 0.4647894736398328, 
                             0.4637894736398328, 0.4637894736398328, 0.463894736576425, 0.463894736576425, 0.463894736576425, 
                             0.462894736576425, 0.462894736576425, 0.4624736840785493, 0.4624736840785493, 0.4624736840785493, 
                             0.4716842106848611, 0.4716842106848611, 0.47373684203924693, 0.47373684203924693, 0.4803684210196165, 
                             0.4803684210196165, 0.4803684210196165, 0.4803684210196165, 0.4803684210196165, 0.4844210526712077, 
                             0.4868947368421097, 0.4877894736842283, 0.4877894736842283, 0.4872105263884713, 0.4872105263884713, 
                             0.48557894736849194, 0.48557894736849194, 0.4791578947371258, 0.4791578947371258, 0.4791578947371258,
                             0.4791578947371258, 0.4661578952849016, 0.4623157894748189, 0.4623157894748189, 0.4623157894748189, 
                             0.4623157894748189, 0.4466842111096475, 0.4466842111096475, '''

        self.fixted_turn = False
        self.turn_control = [0.32668421253874985, 0.29315789630830535, 0.29315789630830535, 0.2844736874861782, 0.2844736874861782, 
                             0.2747368463452887, 0.2747368463452887, 0.26294736355613635, 0.26294736355613635, 0.26294736355613635, 
                             0.26231579232236346, 0.26231579232236346, 0.26231579232236346, 0.2692631579128909, 0.2482105223938823, 
                             0.2482105223938823, 0.2482105223938823, 0.2482105223938823, 0.2492105223938823, 0.2492105223938823, 
                             0.2572105223938823, 0.2572105223938823, 0.2572105223938823, 0.26473683655419844, 0.26473683655419844, 
                             0.26473683655419844, 0.2763157923223635, 0.26031579232236346, 0.26031579232236346, 0.26542104841795267,
                             0.26542104841795267, 0.26794736355613635, 0.26794736355613635, 0.2755263172278326, 0.2755263172278326, 
                             0.28752631722783256, 0.28752631722783256, 0.28752631722783256, 0.2986842089572172, 0.2986842089572172, 
                             0.2986842089572172, 0.2815789417447614, 0.28410526006482106, 0.28410526006482106, 0.2845789417447614, 
                             0.2845789417447614, 0.2870526345283378, 0.2870526345283378, 0.2890000048597772, 0.2890000048597772, 
                             0.2890000048597772, 0.27694736355613636, 0.27694736355613636, 0.27694736355613636, 0.28147368668439754, 
                             0.2798947407966006, 0.2798947407966006, 0.27784210235679396, 0.27784210235679396, 0.27784210235679396, 
                             0.2793157923223635, 0.2879473635561363, 0.2879473635561363, 0.28015789917973155, 0.28015789917973155, 
                             0.2853157884219598, 0.2853157884219598, 0.2900526285146232, 0.2900526285146232, 0.2900526285146232, 
                             0.29531578912997114, 0.29531578912997114, 0.30210526435419893, 0.31242105104858753, 0.32178947051596035, 
                             0.32178947051596035, 0.33515789620870134, 0.33515789620870134, 0.3505789460214959, 0.3505789460214959, 
                             0.37389473947616164, 0.37389473947616164, 0.3867368433652157, 0.3867368433652157, 0.3867368433652157, 
                             0.4149473684871308, 0.4149473684871308, 0.4149473684871308, 0.4262631571331362, 0.4262631571331362, 
                             0.4262631571331362, 0.4532631585937632, 0.4532631585937632, 0.4702105261179069, 0.4702105261179069, 
                             0.477894736576425, 0.477894736576425, 0.48505263152289, 0.48505263152289, 0.4906842106848611, 0.4906842106848611, 
                             0.4906842106848611, 0.49831578933500553, 0.49831578933500553, 0.49831578933500553, 0.5004210526712077, 
                             0.5044736842105266, 0.5044736842105266, 0.5100526315789485, 0.5100526315789485, 0.5126315789803972, 
                             0.5126315789803972, 0.5121578946575993, 0.5121578946575993, 0.5106842106650216, 0.5106842106650216, 
                             0.5106842106650216, 0.5147368419864065, 0.5147368419864065, 0.5147368419864065, 0.5173684213300703,
                             0.5190000000000999, 0.5190000000000999, 0.5190000000000999, 0.5195789472232585, 0.5169473684772701,
                             0.5169473684772701, 0.5153684213300703, 0.5153684213300703, 0.5142631579608085, 0.5142631579608085, 
                             0.5132105263158072, 0.5132105263158072, 0.5132105263158072, 0.5132105263158072, 0.5126842106650216, 
                             0.5126842106650216, 0.5126842106650216, 0.5121578946575993, 0.5121578946575993]
        self.turn_control_seq = 0

        self.goal_list = [self.mission23_goal, self.mission45_goal]

        self.vehicle_length = 0.26

        # 저속3니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =0.8 일때( 3km/h) 0.35
        self.lookahead_distance = 0.8

        self.angle_offset = 0
        self.gain = 1.0

        self.is_current_vel = False
        self.is_state = False
        self.current_vel = 0.0

        self.NO_RIGHTLINE = False
        self.NO_LEFTLINE = False 

        self.DETECT_BOTH = False 

        # 20이 기본 세팅 값
        self.CURVE_ENDPOINT = 40    #@손희문 : 이전 20 에서 5으로 수정 > 10으로 늘릴까유>?
        self.curve_endpoint = self.CURVE_ENDPOINT # 코너 끝까지 빠져나올 때까지 속도 유지하기 위함
        
        # self.curve_endpoint = True
        self.curve_start = True
        self.corner_count = 0
        self.control_angle = Twist()

        # 첫 번째 커브인 경우 차선만 보고 이동
        self.first_curve_start = False

        self.stop_flag = False
        
        # 장애물은 2단계로 이루어져 있음. 정적/동적 or 동적/정적
        self.obstacle_stage = 1

        # 그런데, 처음이 정적인지, 동적인지 모르니까 이걸로 처음이 뭐였는지 체크함
        self.obstacle_type = None

        self.prev_obst_point = None
        self.yaw_diff_mission1 = 0

        self.rotary_exit_flag = False

        self.LANE_DRIVE_VEL = 1800 
        self.TURN_VEL = 1500
        self.INPO_VEL = 1200
        self.STRA_VEL = 2200

        self.target_vel = self.LANE_DRIVE_VEL

        self._min_angle = 10000000
        self._max_angle = -10000000

        #pid 변수 
        self.integral = 0 
        self.prev_error = 0 

        # AMCL pose Subscribe
        self.now_pose = None
        self.now_orientation = None
        self.now_covariance = None
        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5

        self.stop_lane_cnt = 0

        self.prev_stop_lane_flag = False
        self.stop_lane_flag = False
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_callback)

        #AMCL pose Publish
        self.republish_amcl= rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
        
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

        self.goal_list_offset_basis_pos = []
        self.offset_seq = 0

        rospy.Subscriber('/stop_lane_information', Int32, self.stop_lane_callback)
        rospy.Subscriber('/GetTrafficLightStatus', GetTrafficLightStatus, self.mission3)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        # self.now_pose.x += self.position_offset_x
        # self.now_pose.y += self.position_offset_y

        self.now_orientation = msg.pose.pose.orientation

        self.now_covariance = msg.pose.covariance

        self.path_pub.publish(self.global_path)

    def lane_callback(self, msg: LaneInformation):
        if self.MISSION == 0: return

        # 직선 = 곡률 반지름이 엄청 커지기에 곡률은 작다straight 보다 작으면
        # 곡선
        self.L_curv = msg.left_gradient
        self.R_curv = msg.right_gradient
        self.L_point = msg.left_lane_points
        self.R_point = msg.right_lane_points

        L7_point = self.L_point[7].x
        R7_point = self.R_point[7].x

        # if not self.first_curve_start:
        #     self.target_vel = self.LANE_DRIVE_VEL

        #     if np.pi * 17 / 18 <= abs(self.get_yaw_from_orientation(self.now_orientation)):
        #         self.first_curve_start = True
        #         # self.stop_flag=True
        #         # self.stop()
                
        return

        # 1/25 19:46 ==> 4 였음
        reference_quat = self.goal_list[int(self.MISSION > 2)][
            # self.sequence 보다 뒤에 있는 점 중에 lookahead_distacne 랑 비슷한 친구를 더하는게 맞는거
            min(self.sequence + 10, len(self.goal_list[int(self.MISSION > 2)]) - 1)
        ].target_pose.pose.orientation  # 이전 4

        reference_yaw = self.get_yaw_from_orientation(reference_quat)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        
        # @아침수정 벡터로 변환해서 내적으로 각도 차이 구하고, 외적으로는 좌회전 우회전 방향 결정 
        ref_vec = self.direction_vector(reference_yaw)
        vec = self.direction_vector(yaw)
        self.yaw_diff_mission1 = np.arccos(np.dot(ref_vec,vec)) 
        ##print("차선인식에서 받는 ref quat 과 현재 pose 각도 차이 ", self.yaw_diff_mission1)
        # self.yaw_diff_mission1 = abs(yaw - reference_yaw)

        ##print(self.yaw_diff_mission1)
        
        # 검은 바탕일 때
        self.NO_LEFTLINE = len(np.unique([pt.x for pt in self.L_point])) < 4
        self.NO_RIGHTLINE = len(np.unique([pt.y for pt in self.R_point])) < 4

        # 차선이 양쪽 다 있는 경우
        self.DETECT_BOTH = not (self.NO_LEFTLINE or self.NO_RIGHTLINE)

        MID_R_POINT = 498
        MID_L_POINT = 190
        MID_POINT = 342
        mid_point = (L7_point + R7_point) / 2

        if abs(self.yaw_diff_mission1) > 0.3:  # way point 상 회전해야 하는 경우
            self.target_vel = self.TURN_VEL
            
            if self.curve_start:
                # # #print("커브구간 진입  ")
                self.curve_start = False
            
            self.angle_offset = 0
            if np.cross(vec, ref_vec) > 0: #좌회전인 경우
                #print('좌회전임')
                if self.DETECT_BOTH == True: 
                    self.angle_offset = (MID_POINT - mid_point) 
                elif self.NO_LEFTLINE == True and self.NO_RIGHTLINE == False: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                    self.angle_offset = (L7_point - MID_L_POINT)  
                    pass
                    ##print("차선에서 벗어난 정도(왼) ", self.angle_offset)

                else:
                        pass
                    ##print("차선에서 벗어난 정도(정) ", self.angle_offset)

            else:  #우회전인 경우 
                #print("우회전 중 ")
                if self.DETECT_BOTH == True: 
                    self.angle_offset = (MID_POINT - mid_point) 

                elif self.NO_LEFTLINE == True and self.NO_RIGHTLINE == False: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                    self.angle_offset = (R7_point - MID_R_POINT)  #양수면 좌회전 조향 , 음수면 우회전 조향 

                    ##print("차선에서 벗어난 정도(오) ", self.angle_offset)

                else:
                    pass
                    ##print("차선에서 벗어난 정도(정) ", self.angle_offset)

            self.angle_offset = (self.angle_offset / 120) * np.pi / 8


        else: # 직진하는 경우

            if  self.curve_endpoint != 0 and not self.curve_start :  # Ld값 짧은상태=코너 주행중이었다면, 2번 속도 증가무시
                ##print("좌회전이나 우회전의 꼬리부분 ")

                self.target_vel = self.TURN_VEL
                # self.curve_endpoint = False회
                self.curve_endpoint -= 1
                self.angle_offset = 0
                if np.cross(vec, ref_vec) > 0: #좌회전임
                    # #print('좌회전임')
                    if self.DETECT_BOTH == True: 
                        self.angle_offset = (MID_POINT - mid_point) 
                    elif self.NO_RIGHTLINE == True and self.NO_LEFTLINE == False: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                        self.angle_offset = (L7_point - MID_L_POINT)  

                        ##print("차선에서 벗어난 정도(왼) ", self.angle_offset)
                    else: 
                        pass

                else:  #우회전임 
                    ##print("우회전 중 ")
                    if self.DETECT_BOTH == True: 
                        self.angle_offset = (MID_POINT - mid_point) 

                    elif self.NO_LEFTLINE == True and self.NO_RIGHTLINE == False: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                        self.angle_offset = (R7_point - MID_R_POINT)  #양수면 좌회전 조향 , 음수면 우회전 조향 

                        ##print("차선에서 벗어난 정도(오) ", self.angle_offset)

                    else:
                        pass

                self.angle_offset = (self.angle_offset / 120) * np.pi / 8


            else:
                # #print("고속 직진 중 ")
                self.target_vel = self.STRA_VEL
                # self.curve_endpoint = True
                self.curve_endpoint = self.CURVE_ENDPOINT
                # 8km/h 일 때 twist 2.0
                self.curve_start=True
   
                if self.DETECT_BOTH == True: 
                     self.angle_offset = (MID_POINT - mid_point) 
                else:     
                    if self.NO_LEFTLINE == True or self.NO_RIGHTLINE == True:  # 오른쪽이나 왼쪽 차선 없음
                        if self.NO_LEFTLINE: #왼쪽 차선만 없는 경우 오른쪽 보고 간다 .
                            self.angle_offset = 0  #양수면 좌회전 조향 , 음수면 우회전 조향 
                            ##print("차선에서 벗어난 정도(오) ", self.angle_offset)

                        elif self.NO_RIGHTLINE == True: # @1/27 오른쪽 차선만 없는 경우 왼쪽만 보고 간다 > d
                            self.angle_offset = 0 
                            ##print("차선에서 벗어난 정도(왼) ", self.angle_offset)

                self.angle_offset = (self.angle_offset / 120) * np.pi / 100

    def ld_callback(self, data: Odometry):
        # # # # #print(self.MISSION)
        # R_curv 가 0인 경우 대책을 세워야 함
        v = data.twist.twist.linear.x
        # #print('속도', v)

        # 오른쪽 차선의 기울기
        Curv = abs(self.R_curv)

        # 만약 로터리 진입했다면 장애물 자랑 속도 맞추기(로터리 끝나느 waypoint까지)
        if self.car_flag:
            v = 0.3

        # pose 가 많이 틀어졌을 경우 
        if v < 1. :
            self.lookahead_distanc = 0.8
        
        if v < 1.5  :  # 코너를 돌 때 
            #self.lookahead_distance = 0.8840000000000001
            # #print('turning')
            self.lookahead_distance = 1.3
        else: # 직진하는 경우
            #self.lookahead_distance = 0.8840000000000001
            self.lookahead_distance = 1.5

        if self.MISSION == 0:
            # #print('mission0 hi?')
            self.mission0()
        
        else: self.run()

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )

        return euler[2]  # yaw

    def mission0(self): # MISSION0 슬램 구간 탈출
        if not self.now_pose:
            return
        
        # #print('self.now_pose live')

        if self.client.get_state() == GoalStatus.SUCCEEDED:
            self.client.cancel_goal()
            # TEB 종료
            os.system('rosnode kill /throttle_interpolator')
            # os.system('rosnode kill /move_base')
            self.MISSION = 1

        elif self.client.get_state() != GoalStatus.ACTIVE:
            self.client.send_goal(self.slam_goal)

    def rotation_matrix(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ], dtype=np.float32)

    def mission1(self, msg: LidarObstacleInfoArray): # 장애물 구간
        if True or (self.MISSION != 1 and self.MISSION != 2): return
        if not self.now_pose: return

        obstacle_infos = msg.obstacle_infos
        if not len(obstacle_infos):
            self.avoid_flag = False
            self.stop_flag = False

            self.prev_obstacle_pt = None
            self.obstacle_judge_cnt = 0
            self.obstacle_type = None
            return

        # 나랑 가장 가까운 장애물을 가지고 판단
        dists = np.array([np.hypot(info.obst_x, info.obst_y) for info in obstacle_infos])
        chk_obstacle_idx = np.argmin(dists)

        dist = max(0., dists[chk_obstacle_idx] - 0.21)
        
        # 거리가 가까우면 일단 정지함
        # 장애물 타입이 결정되지 않은 경우
        if not self.obstacle_type and dist < 0.8:
            self.stop()
            self.stop_flag = True
            # print('hi?')

            if self.prev_obstacle_pt is None:
                self.prev_obstacle_pt = obstacle_infos[chk_obstacle_idx]

            else:
                self.obstacle_judge_cnt += 1
                if self.obstacle_judge_cnt < 5: return

                now_v = np.array([obstacle_infos[chk_obstacle_idx].obst_x, obstacle_infos[chk_obstacle_idx].obst_y])
                now_norm = np.linalg.norm(now_v)
                now_v /= now_norm
                
                prev_v = np.array([self.prev_obstacle_pt.obst_x, self.prev_obstacle_pt.obst_y])
                prev_norm = np.linalg.norm(prev_v)
                prev_v /= prev_norm
                
                angle = np.arccos(np.dot(now_v, prev_v))
                print('각', angle)

                self.obstacle_type = 's' if abs(angle) < 0.02 else 'd'

        elif self.obstacle_type == 's': # 정적 장애물인 경우
            print('정적')

            # 지금은 뭐지
            # 그냥 라인 보고 가잖아
            # 그래서 중간에 꼬일수도 있으니까 
            # obstacle_avoidance 리스트를 만들어서
            # 이 친구를 다 돌 때까지는 lane_detection 으로 안감
            # 즉, pure pursuit 으로 간다 라고 하면 될 듯

            self.stop_flag = True
            self.stop()
            return

            # 특정 영역 밖이면 그냥 지나가기
            if obstacle_infos[chk_obstacle_idx].obst_x < -0.2 or obstacle_infos[chk_obstacle_idx].obst_x > 0.2:
                print('음?')
                self.stop_flag = False
                return
            
            else:
                print('음??')
                # 회피기동
                pass
        
        elif self.obstacle_type == 'd': # 동적 장애물인 경우
            print('동적')
            # 장애물의 x 좌표가 0보다 큰 경우 ==> 차선을 탈출했다고 판정하고 출발
            if obstacle_infos[chk_obstacle_idx].obst_x > 0.25:
                self.stop_flag = False
                return

            # if self.obstacle_judge_cnt == 2:
            #     self.stop_flag = False
            #     return
            
            # 조건을 바꾸긴 해야 함
            if 0.1 < obstacle_infos[chk_obstacle_idx].obst_y or obstacle_infos[chk_obstacle_idx].obst_y < -0.2:
                self.stop_flag = False
                return
            
            self.stop_flag = True
            self.stop()

                # # 재정지 해야하는 경우임
                # # 장애물의 x 좌표가 -0.8 인 경우 그리고, 장애물의 좌표가 0.1 보다 큰 경우 거리가 아직 1.3 보다 작은 경우는 정지
                # elif obstacle_infos[chk_obstacle_idx].obst_x > -0.8 and obstacle_infos[chk_obstacle_idx].obst_y > 0.1 and dist < 0.95:
                #     self.stop_flag = True
                #     self.stop()

            # 지금은 정적 장애물의 경우에만 따지는 걸로
            # 그리고 동적 장애물은 이전 위치와 현재 위치가 달라지면 알아보는 걸로

    def mission2(self, msg: RotaryArray):
        self.move_car = msg.moving_cars
        self.min_dis = min(self.move_car,key = lambda x: x.dis)

    def get_pos(self):
        return self.goal_list[int(self.MISSION > 2)][self.sequence].target_pose.pose.position

    def stop_lane_callback(self, msg: Int32):
        self.prev_stop_lane_flag = self.stop_lane_flag
        self.stop_lane_flag = msg.data > 70000

        if self.prev_stop_lane_flag and not self.stop_lane_flag:
            # self.republish_initialpose() #이때 amcl 좌표 보정해버림 > 이 함수가 실행되어 Pose 재발행 
            self.stop_lane_cnt += 1
            
    def mission3(self, msg: GetTrafficLightStatus):
        if not self.rotary_exit_flag: return

        #print('정지선', self.stop_lane_flag)
        #print('신호', msg.trafficLightStatus < 16)

        if msg.trafficLightIndex != 'SN000005':
            return

        # 정지선이고 좌회전 신호(33), 직진 신호(16)가 아니라면
        if self.stop_lane_cnt >= 5 and msg.trafficLightStatus < 16:
            print('신호등 정지')
            self.stop()
            self.stop_flag = True

        else:
            # @@@@@@@@@@@@@@@@@@@@@@@위험!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #self.rotary_exit_flag = False
            pass

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
        if not self.now_pose: return
        
        if not self.rotary_exit_flag and self.dist(self.goal_list[int(self.MISSION > 2)][self.sequence].target_pose.pose.position) < self.lookahead_distance:
            if self.sequence == len(self.goal_list[int(self.MISSION > 2)]) - 1:
                # 지금은 거리로 하고 있는데, 부정확해보임 따라서, 정지선 판단 등을 넣어야 할듯?

                # 모든 미션이 끝난 경우
                if self.MISSION > 2:
                    self.stop_flag = True
                    self.stop()
                    return
                
                # 장애물이 막 끝난 경우 == 로터리
                elif self.prev_stop_lane_flag and not self.stop_lane_flag:
                    # 근데 이게 정상적으로 라인에 들어왔을 때 잘 동작함
                    # 정지선이 있다면 로터리 코드 실행
                   #if self.stop_lane_flag:
                        # 로터리 언저리에 차량이 있으면 정지
                    if 0 < self.min_dis.dis < 1.5:
                        self.stop_flag = True 
                        self.stop() 
                        
                    else: # 없으면 출발
                        #print('else')

                        self.target_vel = self.TURN_VEL # 이거 다음은 회전이니까
                        self.rotary_exit_flag = True
                        self.sequence = 0

                        self.stop_flag = False
                        self.MISSION = 3
                        return
                       
            else :
                for seq in range(self.sequence, len(self.goal_list[int(self.MISSION > 2)])):
                    if self.dist(self.goal_list[int(self.MISSION > 2)][seq].target_pose.pose.position) < self.lookahead_distance:
                        self.sequence = seq
        
        # 차량 좌표계는 (차량 x축) = (일반 y축), (차량 y축) = -(일반 x축)
        dx = self.goal_list[int(self.MISSION > 2)][self.sequence].target_pose.pose.position.x - self.now_pose.x
        dy = self.goal_list[int(self.MISSION > 2)][self.sequence].target_pose.pose.position.y - self.now_pose.y 

        # #print('dy', dy, 'dx', dx)

        # Pure Pursuit 알고리즘 적용
        # 목표지점과 현재 지점 간 각도
        vec_to_target = np.array([dx, dy])
        vec_to_target /= np.linalg.norm(vec_to_target)

        yaw = self.get_yaw_from_orientation(self.now_orientation) #orientation is quat
        
        # 방향 벡터
        pose_vec = self.direction_vector(yaw)

        # 목표 지점과 나의 pose의 차이를 계산 > 내적으로 크기 계산 
        direction = np.cross(pose_vec,vec_to_target)
        if direction>0:
            angle_difference = -np.arccos(pose_vec @ vec_to_target)
        else: 
            angle_difference = np.arccos(pose_vec @ vec_to_target)
        # 이거는 각도 정규화
        
        ##print(f"실제 계산되는 각도 차이 : {angle_difference}")

        # 원래 4.93 , 1.8 중 뭐지?

        self.gain = 1
        
        # TILT = abs(angle_difference) #틀어진 정도 

        # if self.target_vel == self.TURN_VEL: #코너 상황들어갈 때
        #     pass
                     
        # elif self.target_vel != self.LANE_DRIVE_VEL:
        #     print(self.target_vel)
        #     if TILT < 0.1 :
        #         print(f"똑바른 직선")
        #         self.corner_count = 0
        #         self.target_vel = self.STRA_VEL
            
        #     elif self.corner_count > 4:
        #         self.target_vel = self.TURN_VEL 
        #         print(f"코너 끝나고 수평 안맞을 때 gain값")

        #     elif not self.rotary_exit_flag:
        #         self.target_vel = self.INPO_VEL
        #         print("직선에서 어긋났을때 = tgt vel 2.0일때 gain값")
        
        steering_angle = self.gain * np.arctan2(2.0 * self.vehicle_length * np.sin(angle_difference) / self.lookahead_distance, 1.0)
        # #print("pure pursuit으로 계산되는 steering angle", steering_angle)
        steering_angle = self.mapping(steering_angle) + self.angle_offset

        if self.target_vel == self.LANE_DRIVE_VEL:
            e = 0.00000000001
            some_value = 1 / ((self.R_curv + e) * 2.5) # 살짝 키운값

            MID_L_POINT = 190
            MID_POINT = 342

            l7_pt = self.L_point[7].x
            r7_pt = self.R_point[7].x

            m = (l7_pt + r7_pt) // 2

            if self.stop_lane_cnt == 4 and ((np.pi * 17 / 18 <= abs(self.get_yaw_from_orientation(self.now_orientation)) and abs(self.L_curv) < 0.1) or self.fixted_turn) and self.turn_control_seq < len(self.turn_control):
                    print('왼쪽으로 꺾어야 함!', self.turn_control_seq)
                    self.fixted_turn = True
                    steering_angle = self.turn_control[self.turn_control_seq]
                    self.turn_control_seq += 1

            elif self.stop_lane_cnt == 5:
                print('self.stop_lane_cnt == 5')
                self.stop()
                self.stop_flag = True

            else:
                steering_angle = self.mapping(some_value, -20, 20, 1, 0)
                steering_angle -= (MID_POINT - m) / 1000
            
            self.target_vel = self.LANE_DRIVE_VEL
        
        ##print(" 조향 각 크기: ", steer )
        if not self.stop_flag:
            # #print(self.target_vel, steering_angle)
            self.vel_pub.publish(Float64(data = self.target_vel))
            self.steer_pub.publish(Float64(data = steering_angle))

    def mapping(self, value, from_min=-0.20992659995162515, from_max=0.20992659995162515, to_min=0, to_max=1):
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
 
    def republish_initialpose(self):
        # 만약에 정지선을 발견했다면
        # 미리 저장된 그 정보를 바탕으로 전체적으로 시퀀스를 조정해야 함
        # 1. 일단 goal_list 에 있는 모든 좌표에 offset을 +- 해줘야 함
        basis_x = self.now_pose.x
        basis_y = self.now_pose.y

        # 1-1. offset 구하기
        offset_x = self.goal_list_offset_basis_pos[self.offset_seq].pose.pose.position.x - basis_x
        offset_y = self.goal_list_offset_basis_pos[self.offset_seq].pose.pose.position.y - basis_y
        
        print(offset_x, offset_y)

        # 1-2. 모든 goal_list에 있는 모든 좌표에 offset 적용
        self.global_path = Path()
        self.global_path.header.frame_id = 'map'
        
        for pos in range(len(self.goal_list[0])):
            self.goal_list[0][pos].target_pose.pose.position.x += offset_x
            self.goal_list[0][pos].target_pose.pose.position.y += offset_y

            pt = PoseStamped()
            pt.header.frame_id = 'map'
            pt.pose.position.x = self.goal_list[0][pos].target_pose.pose.position.x
            pt.pose.position.y = self.goal_list[0][pos].target_pose.pose.position.y
            pt.pose.orientation.z = self.goal_list[0][pos].target_pose.pose.orientation.z
            pt.pose.orientation.w = self.goal_list[0][pos].target_pose.pose.orientation.w
            
            self.global_path.poses.append(pt)

        for pos in range(len(self.goal_list[1])):
            self.goal_list[1][pos].target_pose.pose.position.x += offset_x
            self.goal_list[1][pos].target_pose.pose.position.y += offset_y

            pt = PoseStamped()
            pt.header.frame_id = 'map'
            pt.pose.position.x = self.goal_list[1][pos].target_pose.pose.position.x
            pt.pose.position.y = self.goal_list[1][pos].target_pose.pose.position.y
            pt.pose.orientation.z = self.goal_list[1][pos].target_pose.pose.orientation.z
            pt.pose.orientation.w = self.goal_list[1][pos].target_pose.pose.orientation.w
            
            self.global_path.poses.append(pt)

        self.path_pub.publish(self.global_path)

        # 2. 그리고 현재 내 위치를 기반으로 다시 시작 시퀀스를 찾아야 함 ==> 왜냐 self.lookahead 때문에 sequence 가 현재 내 위치 언저리를 가리키지 않은 상태이기 때문임
        # 일단 순차적으로 탐색하면서 내 방향이랑 비슷한 시퀀스 및 좌표를 가진 seq를 가져와야 하는 부분임
        # 기억에 mission23에 대해서만
        self.sequence = np.argmin(np.array([np.hypot(pos.target_pose.pose.position.x - basis_x, pos.target_pose.pose.position.y - basis_y) for pos in self.goal_list[0]]))
        for seq in range(self.sequence, len(self.goal_list[0])):
            if self.dist(self.goal_list[0][seq].target_pose.pose.position) < self.lookahead_distance:
                self.sequence = seq

        # # 위치 조정 후 발행 
        # self.republish_amcl.publish(init_pose)
        
if __name__ == "__main__":

    nc = Total()
    rospy.spin()
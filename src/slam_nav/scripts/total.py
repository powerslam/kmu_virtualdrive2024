#! /usr/bin/env python3
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 제출할 때 상대경로로 바꿔서 제출해야 함!
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import tf
import rospy
import pickle
import actionlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt

import os

import rospkg

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
        self.return_ok = False
        self.traffic_sign = None
        self.prev_R_curv = None

        # 현재 수행해야 할 미션이 무엇인지 가리키는 함수
        self.MISSION = 0

        self.TEB_Planner = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.TEB_Planner.wait_for_server()

        self.vel_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        self.path_pub = rospy.Publisher('/global_pah', Path, queue_size=1)
        
        self.stop_data = Float64(data=0)

        self.sequence = 0  # 받아온 좌표모음의 index
        self.obstacle_point = -1

        # SLAM 구간용 코드(mission 1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.slam_goal = MoveBaseGoal()
        self.slam_goal.target_pose.header.frame_id = 'map'
        self.slam_goal.target_pose.pose.position.x = 17.71121201250607
        self.slam_goal.target_pose.pose.position.y = -9.909904223816536
        self.slam_goal.target_pose.pose.orientation.z = 0
        self.slam_goal.target_pose.pose.orientation.w = 1

        self.global_path = Path()
        self.global_path.header.frame_id = 'map'

        self.position_offset_x = 0
        self.position_offset_y = 0

        self.direction = 0 
        self.dy_flag = False
        
        self.prev_yaw = 0
        self.test_minmax = []  #TILT
        self.obst_cnt = 0
        self.obstacle_avoidance = [0.0,
0.4865273584114077,
0.47986819773439054,
0.4542690233525442,
0.33615203973489094,
0.2009182913506854,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.085782213956042,
0.0752309448837245,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.22733211740001383,
0.2665441809425161,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.3467320673790381,
0.7033825668293232,
0.864946201478273,
0.9692878339427365,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
0.4543684180381112,
0.41847368015247405,
0.38105263733847544,
0.38105263733847544,
0.1653684211217941,
0.1653684211217941,
0.1653684211217941,
0.20442104630898358,
0.16147368706963683,
0.16147368706963683,
0.13363157828579042,
0.13363157828579042,
0.1124705830188664,
0.1124705830188664,
0.1124705830188664,
0.1588235248287504,
0.1588235248287504,
0.1588235248287504,
0.18200000451388643,
0.20094737133457374,
0.20094737133457374,
0.2247368487048681,
0.2551052676109338,
0.2875263162253969,
0.2875263162253969,
0.2875263162253969,
0.2875263162253969,
0.3409473663941583,
0.3409473663941583,
0.39121052420464814,
0.39121052420464814,
0.39121052420464814,
0.39121052420464814,
0.39121052420464814,
0.41326315686744763,
0.45315789658742933,
0.45315789658742933,
0.45315789658742933,
0.48689473698128616,
0.48689473698128616,
0.5098421050982213,
0.5098421050982213,
0.5170000000001,
0.5170000000001,
0.518578947368492,
0.518578947368492,
0.5176315786700382,
0.5176315786700382,
0.5161578946081032,
0.5141578946081032,
0.5131578946081032,
0.5131578946081032,
0.5121578946081032,
0.5121578946081032,
0.5121578946081032,
0.5121578946081032,
0.5132105263884713,
0.5132105263884713,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5106315786700382,
0.5101052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5096315786700382,
0.5096315786700382,
0.5096315786700382,
0.5096315786700382,
0.5081052630588911,
0.5081052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.508578947368492,
0.508578947368492,
0.508578947368492,
0.5096315786700382,
0.5096315786700382,
0.5096315786700382,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5091052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.5101052630588911,
0.509578947368492,
0.509578947368492,
0.509578947368492,
0.509578947368492,
0.509578947368492,
0.5101052630588911
]

        # 로터리 및 좌회전 미션(로터리 앞 정지선 ~ 골 지점까지)
        self.goal_list = []

        rospack = rospkg.RosPack()
        slam_nav_pkg_path = rospack.get_path('slam_nav')
        with open(slam_nav_pkg_path + "/scripts/mission45.pkl", "rb") as file:
            pt_list = pickle.load(file)

            for _pt in pt_list:
                pt = PoseStamped()
                pt.header.frame_id = "map"
                pt.pose.position.x = _pt.position.x
                pt.pose.position.y = _pt.position.y
                pt.pose.orientation.z = _pt.orientation.z
                pt.pose.orientation.w = _pt.orientation.w

                self.goal_list.append(pt)

                pt = PoseStamped()
                pt.header.frame_id = 'map'
                pt.pose.position.x = _pt.position.x
                pt.pose.position.y = _pt.position.y
                pt.pose.orientation.z = _pt.orientation.z
                pt.pose.orientation.w = _pt.orientation.w
                
                self.global_path.poses.append(pt)

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
                             0.477894736576425, 0.477894736576425, 0.48505263152289, 0.48505263152289, 0.4906842106848611]
        self.turn_control_seq = 0

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

        self.obstacle_judge_cnt = 0

        self.prev_obstacle_pt = None
        self.yaw_diff_mission1 = 0

        self.rotary_exit_flag = False

        self.LANE_DRIVE_VEL = 1800
        self.LANE_DRIVE_LAST_VEL = 1201
        self.OBST_VEL = 602
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

        if self.MISSION < 4: return

        # print('pure pursuit 으로 주행해요')

        # 1/25 19:46 ==> 4 였음
        reference_quat = self.goal_list[
            # self.sequence 보다 뒤에 있는 점 중에 lookahead_distacne 랑 비슷한 친구를 더하는게 맞는거
            min(self.sequence + 3 , len(self.goal_list) - 1)
        ].pose.orientation  # 이전 4

        reference_yaw = self.get_yaw_from_orientation(reference_quat)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        
        ref_vec = self.direction_vector(reference_yaw)
        vec = self.direction_vector(yaw)
        self.yaw_diff_mission1 = np.arccos(np.clip(np.dot(ref_vec,vec), -1., 1.)) 
        
        # 검은 바탕일 때
        self.NO_LEFTLINE = len(np.unique([pt.x for pt in self.L_point])) == 1
        self.NO_RIGHTLINE = len(np.unique([pt.x for pt in self.R_point])) == 1
        self.DETECT_BOTH = not (self.NO_LEFTLINE or self.NO_RIGHTLINE)

        MID_R_POINT = 498
        MID_L_POINT = 190
        MID_POINT = 342
        mid_point = (L7_point + R7_point) / 2

        self.angle_offset = (MID_POINT - mid_point) if self.DETECT_BOTH else 0

        if abs(self.yaw_diff_mission1) > 0.3:  # way point 상 회전해야 하는 경우
            self.target_vel = self.TURN_VEL
            
            if self.curve_start:
                # # #print("커브구간 진입  ")
                self.curve_start = False
            
            #self.angle_offset = (self.angle_offset / 120) * np.pi / 40

        else: # 직진하는 경우
            if self.curve_endpoint > 0 and not self.curve_start :  # Ld값 짧은상태=코너 주행중이었다면, 2번 속도 증가무시
                print("좌회전이나 우회전의 꼬리부분 ")

                self.target_vel = self.TURN_VEL
                self.curve_endpoint -= 1
                #self.angle_offset = (self.angle_offset / 120) * np.pi / 40

            else:
                # #print("고속 직진 중 ")
                self.target_vel = self.STRA_VEL
                self.curve_endpoint = self.CURVE_ENDPOINT # 고정된 endpoint 초기화
                self.curve_start=True
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
            self.lookahead_distance = 0.8
        
        if v < 1.5  :  # 코너를 돌 때 
            #self.lookahead_distance = 0.8840000000000001
            # #print('turning')
            self.lookahead_distance = 1.3

        else: # 직진하는 경우
            #self.lookahead_distance = 0.8840000000000001
            self.lookahead_distance = 1.5

        if self.target_vel == self.LANE_DRIVE_VEL:
            self.lookahead_distance = 1.2

        # pure pursuit 조작
        # 장애물에서 속도가 너무 빠르니까 self.OBST_VEL 을 만들어서 조작함
        # 근데, 속도에 따라 ld 가 달라져서 이런식으로 조건문을 만듬
        # 주행시켜보니까 obstacle_point 끝나기도 전에 자꾸 똑같은 객체보고 경로를 생성하는 문제가 있어서
        # mission1 함수 상단에 obstacle_point > 0 이면 return 하는 조건을 만들어줌
        # 그리고, obstpoint 가 끝나고 회피 시작 시 x 좌표 및 방향이 똑같으면 그제서야 lane 만 보는 주행을 하는 것으로
        # 합의 봄 ==> 이견 있으면 적길 바람
        elif self.target_vel == self.OBST_VEL:
            self.lookahead_distance = 0.6

        if self.MISSION == 0: self.mission0()
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

    # 장애물 구간의 문제점
    # 1. 회피경로 주행 중 또 장애물을 못 봄 => 
    # 2. 지금 장애물은 가까운 거 기준이라서 또 있는 장애물을 못 봄 => y 좌표 가지고 해결 가능 => y가 너무 가까우면 제외하면 되니까

    def mission1(self, msg: LidarObstacleInfoArray): # 장애물 구간
        if self.MISSION != 1 and self.MISSION != 4: return
        if not self.now_pose: return

        obstacle_infos = msg.obstacle_infos
        if not len(obstacle_infos): 
            if self.obstacle_type == 's':
                self.return_ok = True

            return
        
        if self.obstacle_point > 0: return

        # 나랑 가장 가까운 장애물을 가지고 판단
        dists = np.array([np.hypot(info.obst_x, info.obst_y) for info in obstacle_infos])
        chk_obstacle_idx = np.argmin(dists)

        dist = max(0., dists[chk_obstacle_idx] - 0.21)
        info = obstacle_infos[chk_obstacle_idx]

        avoid_offset = 0.314

        yaw = abs(self.get_yaw_from_orientation(self.now_orientation)) #orientation is quat

        if yaw < np.pi / 4:
            print("위로가는 방향 ")
            self.direction = 1 
        elif yaw < 2.6:
            print( "왼쪽 가는 방향 ")
            self.direction = 2 
        else:
            print(" 아래로 가는 방향 ")
            self.direction = 3 
        

        # 거리가 가까우면 일단 정지함
        # 장애물 타입이 결정되지 않은 경우
        if not self.obstacle_type and dist < 1.1:
            self.stop()
            self.stop_flag = True
            self.prev_yaw = yaw

            self.obstacle_judge_cnt += 1
            if self.obstacle_judge_cnt < 8: 
                self.prev_obstacle_pt = obstacle_infos[chk_obstacle_idx]
                return

            now_v = np.array([obstacle_infos[chk_obstacle_idx].obst_x, obstacle_infos[chk_obstacle_idx].obst_y])
            now_norm = np.linalg.norm(now_v)
            now_v /= now_norm
            
            prev_v = np.array([self.prev_obstacle_pt.obst_x, self.prev_obstacle_pt.obst_y])
            prev_norm = np.linalg.norm(prev_v)
            prev_v /= prev_norm

            angle = np.arccos(np.clip(np.dot(now_v, prev_v), -1., 1.))
            print('각도', abs(angle))
            self.obstacle_type = 's' if abs(angle) < 0.1 else 'd'

        elif self.obstacle_type == 's': # 정적 장애물인 경우
            self.obst_cnt += 1
            print('정적')

            if self.obstacle_point == -1:
                # 나는 지금 항상 1차선에서 본다고 생각하고 첫 각도를 30도로 줬는데, 거리나 목표 지점의 각도 차이에 의해서
                # np.pi / 6 -> ?? 으로 변경되어야 함

                # now_yaw = self.get_yaw_from_orientation(self.now_orientation)
                # target_yaw = np.arctan2(np.clip(x - now_yaw, -np.pi, np.pi))

                self.move_left = [
                    0, np.pi / 6, 0
                ]

                self.move_right = [
                    0, -np.pi / 6, 0
                ]

                self.obstacle_point = 6
                # [
                #     0, np.pi / 9.5, np.pi / 6, np.pi / 9.5, 0,
                #     -np.pi / 9.5, -np.pi / 6, -np.pi / 9.5, 0
                # ]
            
                self.target_vel = self.OBST_VEL
                # self.obstacle_point = len(self.target_yaw)
                self.stop_flag = False

            
            # self.target_vel = self.OBST_VEL
            # if self.direction == 1 :
            #     dy_ob_x = info.obst_y + self.now_pose.x # amcl상에서 장애물의 x좌표
            #     dy_ob_y = -info.obst_x + self.now_pose.y # amcl상에서 장애물의 y좌표
            # elif self.direction == 2: 
            #     dy_ob_x = self.now_pose.x # amcl상에서 장애물의 x좌표
            #     dy_ob_y = info.obst_y + self.now_pose.y # amcl상에서 장애물의 y좌표
            # else: #@TODO : 미션 5에서 나오는 경우 , 그냥 장애물좌표 생성 안시키고 무시해야함 
            #     pass 
            # self.stop_flag  = True
            # self.stop()
            # # 차선 변경 실시
            
            # # 장애물 옆으로 회피
            # if self.obst_cnt >= 2 or (self.prev_yaw - yaw) > 5*np.pi / 18 :  #10도 이상 차이가 나는 경우 avoid offset값 줄임  (self.prev_yaw - yaw) > 5*np.pi / 18 and 
            #     print("2 개 연속으로 봄 ")
            #     self.avoid_offset= 0.15

            # if self.direction == 1: #위로가는 방향 
            #     target_x = dy_ob_x 
            #     target_y = dy_ob_y + avoid_offset
            
            # elif self.direction == 2: #왼쪽으로 가는 방향 
            #     target_x = dy_ob_x - avoid_offset  # amcl상에서 이동해야 할 x좌표
            #     target_y = dy_ob_y  # amcl상에서 이동해야 할 y좌표

            #     #y축을 빼야 하는 경우도 있는걸 염두

            # else: # 1차선일 때 #아래로 가는 방향 
            #     #2차선으로 이동
            #     target_x = dy_ob_x - dist       # amcl상에서 이동해야 할 x좌표
            #     target_y = dy_ob_y + avoid_offset      # amcl상에서 이동해야 할 y좌표

            # self.stop_flag  = False
            # self.create_trajectory( target_x, target_y, dist)
        
        elif self.obstacle_type == 'd': # 동적 장애물인 경우
            print('동적')
            # 장애물의 x 좌표가 0보다 큰 경우 ==> 차선을 탈출했다고 판정하고 출발
            if obstacle_infos[chk_obstacle_idx].obst_x > 0.25 or obstacle_infos[chk_obstacle_idx].obst_x < -0.65:
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
        self.min_dis = min(self.move_car, key = lambda x: x.dis)
        
    def get_pos(self):
        return self.goal_list[self.sequence].pose.position

    def stop_lane_callback(self, msg: Int32):
        if self.fixted_turn < 2: return

        self.prev_stop_lane_flag = self.stop_lane_flag
        self.stop_lane_flag = msg.data > 67000

        # 앞에 차량이 지나갈 때 정지선으로 판단할 위험이 있음
        if self.prev_stop_lane_flag and not self.stop_lane_flag:
            if self.MISSION == 4 or self.MISSION == 2:
                self.stop_lane_cnt += 1

        # 로터리 앞 정지선을 발견한 이후에
        if self.stop_lane_cnt == 1 and self.MISSION != 4:
            if abs(self.min_dis.dis) < 0.68: # 도로에 차량이 있는 경우라면 정지
                print('차량이 앞에 있어요')
                self.stop_flag = True
                self.stop()

            else:
                self.target_vel = 1203 # 로터리 스피드
                self.rotary_exit_flag = True
                self.sequence = 4
                self.lookahead_distance = 1.3
                self.stop_flag = False
                self.run()
                return

        if not self.traffic_sign: return
        
        # 근데 차량을 보고 이 친구가 
        if self.stop_lane_cnt > 1:
            print('정지선을 한 번 더 봤어요', self.stop_lane_cnt, self.traffic_sign)
            yaw = self.get_yaw_from_orientation(self.now_orientation)
            # 정방향을 보고, 신호등이 정지 신호면
            if 17 / 18 * np.pi <= abs(yaw) and self.traffic_sign < 16:
                print('신호등을 봤어요')
                self.stop()
                self.stop_flag = True
        
        else: self.stop_flag = False
            
    def mission3(self, msg: GetTrafficLightStatus):
        if not self.rotary_exit_flag: return

        #print('정지선', self.stop_lane_flag)
        #print('신호', msg.trafficLightStatus < 16)

        if msg.trafficLightIndex != 'SN000005':
            return

        self.traffic_sign = msg.trafficLightStatus

    def create_trajectory(self, tgt_x, tgt_y, dist):    
        print("경로가 생성됨 ")
        if self. direction == 2:  # 첫번째 코너를 지난 후 정적장애물
            print("첫번째 코너를 지난 후  정적장애물")
            x = np.array([0,  dist * 0.3,                    dist+0.5,               dist + 1])
            y = np.array([0,           0, self.now_pose.x - tgt_x,                        0])
            

            #print('trajectory x', x)
            #print('trajectory y', y)

            x_scale = x
            y_scale = y

            f = CubicHermiteSpline(x_scale, y_scale, dydx = [0, 0, 0, 0])

            coefficients = f.c.T # 생성된 Cubic의 계수 (coefficients)

            num = int((abs(dist) + 1) / 0.05) 
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

            self.rotated = np.ones(shape=(num, 3))
            self.rotated[:, 0] = x_new
            self.rotated[:, 1] = y_new

            # #print(orientation_new)
            self.sum_coord = self.rotation_matrix(np.pi / 2) #90도 회전
            self.rotated = self.sum_coord @ self.rotated.T
            # (3 x 3) @ (40 x 3).T
            # 3 x 40
            self.rotated /= self.rotated[2]
            self.rotated = self.rotated[:2].T

            self.obstacle_point = 0
            for seq in range(self.sequence + 1, self.sequence + num):
                pose = Pose()

                pose.position.x = self.rotated[self.obstacle_point][0] + self.now_pose.x
                pose.position.y = self.rotated[self.obstacle_point][1] + self.now_pose.y
                
                _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, -orientation_new[self.obstacle_point])

                pose.orientation.x = 0.
                pose.orientation.y = 0.
                pose.orientation.z = qz
                pose.orientation.w = qw

                self.obstacle_avoidance += [pose]
                self.obstacle_point += 1

            plt.figure(figsize = (dist, 0.5))
            plt.plot(self.rotated[:,0], self.rotated[:,1], 'b')
            plt.plot(x_scale, y_scale, 'ro')
            plt.plot(x_new, orientation_new, 'g')
            
            plt.title('Cubic Hermite Spline Interpolation')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        elif self.direction ==3 : # 로터리 지나고 1차선 주행일 때 정적장애물을 만남
            print("로터리 지나고  정적장애물")
            x = np.array([0, 0 + dist * 0.3, abs(tgt_x), abs(tgt_x - 0.5)])
            y = np.array([0, 0       , tgt_y       , tgt_y])

            #print('trajectory x', x)
            #print('trajectory y', y)

            x_scale = x
            y_scale = y

            f = CubicHermiteSpline(x_scale, y_scale, dydx = [0, 0, 0, 0])

            coefficients = f.c.T # 생성된 Cubic의 계수 (coefficients)

            num = int((abs(dist) + 0.5) / 0.05) 
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
                pose = Pose()

                pose.position.x = self.now_pose.x - x_new[self.obstacle_point]
                pose.position.y = self.now_pose.y - y_new[self.obstacle_point]
                
                _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, -orientation_new[self.obstacle_point])

                pose.orientation.x = 0.
                pose.orientation.y = 0.
                pose.orientation.z = qz
                pose.orientation.w = qw

                self.obstacle_avoidance += [pose]
                self.obstacle_point += 1
            
        else : # 첫번째 코너를 지나기 전 정적장애물
            print("첫번째 코너를 지나기 전 정적장애물")
            # x = np.array([self.now_pose.x, self.now_pose.x + dist * 0.3, tgt_x + 0.5, tgt_x + 1, tgt_x + 1.5])
            # y = np.array([self.now_pose.y, self.now_pose.y, tgt_y, self.now_pose.y + 0.16, self.now_pose.y + 0.7])

            x = np.array([self.now_pose.x, self.now_pose.x + dist * 0.3, tgt_x, tgt_x + dist * 0.3, tgt_x + dist, tgt_x + dist * 1.3])
            y = np.array([self.now_pose.y, self.now_pose.y, tgt_y, tgt_y, self.now_pose.y,  self.now_pose.y + 0.2])

            #print('trajectory x', x)
            #print('trajectory y', y)

            x_scale = x
            y_scale = y
            current_yaw= self.get_yaw_from_orientation(self.now_orientation)
            f = CubicHermiteSpline(x_scale, y_scale, dydx = [np.tan(current_yaw), 0, 0, 0, 0, 0])

            coefficients = f.c.T # 생성된 Cubic의 계수 (coefficients)

            num = int((abs(dist) * 1.3) / 0.05) 
            x_new = np.linspace(x_scale[0], x_scale[-1], num) # 경로 상의 절대 x 좌표
            y_new = f(x_new)   #경로상의 절대 y좌표

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
                pose = Pose()

                pose.position.x = x_new[self.obstacle_point]
                pose.position.y = y_new[self.obstacle_point]
                
                _, _, qz, qw = tf.transformations.quaternion_from_euler(0, 0, -orientation_new[self.obstacle_point])

                pose.orientation.x = 0.
                pose.orientation.y = 0.
                pose.orientation.z = qz
                pose.orientation.w = qw

                self.obstacle_avoidance += [pose]
                self.obstacle_point += 1
            
            #print("회피경로 생성 완료 ")

            # plt.figure(figsize = (dist, 0.5))
            # plt.plot(x_new, y_new, 'b')
            # plt.plot(x_scale, y_scale, 'ro')
            # plt.plot(x_new, orientation_new, 'g')
            
            # plt.title('Cubic Hermite Spline Interpolation')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.show()

    def run(self):
        if not self.now_pose: return
        if not self.L_point or not self.R_point: return

        # MISSION 2, 3 구간에 라인으로만 주행
        if self.target_vel == self.LANE_DRIVE_VEL or self.target_vel == self.LANE_DRIVE_LAST_VEL:
            # print('라인으로 주행해요')

            e = 0.00000000001
            some_value = 1 / ((self.R_curv + e) * 2.5) # 살짝 키운값

            MID_L_POINT = 190
            MID_POINT = 342

            l7_pt = self.L_point[7].x
            r7_pt = self.R_point[7].x

            m = (l7_pt + r7_pt) // 2

            steering_angle = self.mapping(some_value, -20, 20, 1, 0)
            if not (self.L_curv < 0 and abs(self.R_curv) > 0.5):
                steering_angle -= (MID_POINT - m) / 1000
            self.target_vel = self.LANE_DRIVE_VEL if self.fixted_turn != 2 else self.LANE_DRIVE_LAST_VEL

            if 0 < self.R_curv < 0.08:
                # print('초기화..')
                self.prev_obstacle_pt = None
                self.obstacle_judge_cnt = 0
                self.obstacle_type = None

                self.obstacle_avoidance = []

                self.target_vel = self.LANE_DRIVE_VEL if self.fixted_turn < 2 else self.TURN_VEL

            if ((np.pi * 17 / 18 <= abs(self.get_yaw_from_orientation(self.now_orientation)) and abs(self.L_curv) < 0.1) or self.fixted_turn == 1) and self.turn_control_seq < len(self.turn_control):
                if 0 < self.R_curv < 0.09:
                    print('이제는 다시 라인을 봐요')
                    self.turn_control_seq = len(self.turn_control)
                    self.fixted_turn = 2
                
                else:
                    print('고정 steer 로 주행해요', self.turn_control_seq)
                    self.fixted_turn = 1
                    steering_angle = self.turn_control[self.turn_control_seq]
                    self.turn_control_seq += 1
                    self.MISSION = 2

            elif self.stop_lane_cnt > 0:
                print('정지선을 봤어요. 이제는 mission45 를 기반으로 주행해요')
                self.target_vel = self.TURN_VEL
                self.MISSION = 3
    
        elif self.obstacle_point >= 0:
            now_yaw = self.get_yaw_from_orientation(self.now_orientation)
            # 현재각이 얼추 그 범위 내로 들어온 경우
            # 2번에 걸쳐서 이 과정이 이루어져야 함
            # obstacle_infos의 길이가 0일 때 

            # 6 5 4
            if self.obstacle_point > 3:
                diff = self.move_left[self.obstacle_point % 3] - now_yaw

                if abs(diff) < np.pi / 36:
                    self.obstacle_point -= 1

            # 3 2 1 ==> 1을 할 차례인데, 0이 나와서 바로 종료되기 때문에 초기 obstacle_point의 값은 -1로 변경
            elif self.return_ok:
                diff = self.move_right[self.obstacle_point % 3] - now_yaw
                if abs(diff) < np.pi / 36:
                    self.obstacle_point -= 1

            else:
                diff = 0.0 - now_yaw

            # M - m error
            steering_angle = self.mapping(np.clip(diff * 1.6, -np.pi / 6, np.pi / 6), -np.pi / 6 , np.pi / 6, 1, 0)
            
            # print(steering_angle)
            if self.obstacle_point == -1:
                if True:
                    self.stop()
                    self.stop_flag = True
                    return

                self.prev_obstacle_pt = None
                self.obstacle_judge_cnt = 0
                self.obstacle_type = None

                self.obstacle_avoidance = []

                self.target_vel = self.LANE_DRIVE_VEL if self.fixted_turn < 2 else self.TURN_VEL
                
                return

        else:
            if self.dist(self.goal_list[self.sequence].pose.position) < self.lookahead_distance:
                if self.sequence == len(self.goal_list) - 1:
                    if self.dist(self.goal_list[self.sequence].pose.position) < 0.1:
                        self.stop_flag = True
                        self.stop()

                else :
                    for seq in range(self.sequence, len(self.goal_list)):
                        if self.dist(self.goal_list[seq].pose.position) < self.lookahead_distance:
                            self.sequence = seq
            calc_pose = self.goal_list[self.sequence].pose
            
            yaw = self.get_yaw_from_orientation(self.now_orientation)
            if abs(yaw) >= 17 / 18 * np.pi:
                # print('미션이 4가 되었어요')
                self.MISSION = 4

            # if self.obstacle_point > 0:
            #     calc_pose = self.obstacle_avoidance[len(self.obstacle_avoidance) - self.obstacle_point]

            #     # print('장애물 남은 시퀀스', self.obstacle_point)
            #     if self.dist(calc_pose.position) < self.lookahead_distance and self.obstacle_point > 0:
            #         self.obstacle_point -= 1

            #     if self.obstacle_point == 0:
            #         if False:
            #             self.stop()
            #             self.stop_flag = True
            #             return

            #         self.prev_obstacle_pt = None
            #         self.obstacle_judge_cnt = 0
            #         self.obstacle_type = None

            #         self.obstacle_avoidance = []

            #         self.target_vel = self.LANE_DRIVE_VEL if self.fixted_turn < 2 else self.TURN_VEL
                    
            #         return

            # 차량 좌표계는 (차량 x축) = (일반 y축), (차량 y축) = -(일반 x축)
            dx = calc_pose.position.x - self.now_pose.x
            dy = calc_pose.position.y - self.now_pose.y 

            # #print('dy', dy, 'dx', dx)

            # Pure Pursuit 알고리즘 적용
            # 목표지점과 현재 지점 간 각도
            e = 0.00000000001
            vec_to_target = np.array([(dx + e), (dy + e)])
            vec_to_target /= np.linalg.norm(vec_to_target)
            yaw = self.get_yaw_from_orientation(self.now_orientation) #orientation is quat
            
            # 방향 벡터
            pose_vec = self.direction_vector(yaw)

            # 목표 지점과 나의 pose의 차이를 계산 > 내적으로 크기 계산 
            direction = np.cross(pose_vec,vec_to_target)

            angle_difference = -1 if direction > 0 else 1
            angle_difference *= np.arccos(np.clip(pose_vec @ vec_to_target, -1., 1.))
            
            # 원래 4.93 , 1.8 중 뭐지?
            self.gain = 1
            
            TILT = abs(angle_difference) #틀어진 정도 

            # print("angle_differecne 출력 ", TILT)
            self.test_minmax.append(TILT)
            # print ("최대최소 갱신", min(self.test_minmax), max(self.test_minmax))
            if self.target_vel == self.TURN_VEL: #코너 상황들어갈 때
                self.gain = self.tilt_mapping(TILT)
                        
            steering_angle = self.gain * np.arctan2(2.0 * self.vehicle_length * np.sin(angle_difference) / self.lookahead_distance, 1.0)
            if self.target_vel != self.LANE_DRIVE_VEL:
                # print('pure 조향각', steering_angle, 'mapping 후', self.mapping(steering_angle + self.angle_offset))
                pass
            
            # #print("pure pursuit으로 계산되는 steering angle", steering_angle)
            steering_angle = self.mapping(steering_angle + self.angle_offset)
            


        if not self.stop_flag:
            # print('움직여요')
            self.vel_pub.publish(Float64(data = self.target_vel))
            self.steer_pub.publish(Float64(data = steering_angle))

    def mapping(self, value, from_min=-0.25992659995162515, from_max=0.25992659995162515, to_min=0, to_max=1):
        value = np.clip(value, from_min, from_max)

        # 선형 변환 수행
        from_range = from_max - from_min
        to_range = to_max - to_min

        # 선형 변환 공식 적용
        mapped_value = ((value - from_min) / from_range) * to_range + to_min

        return mapped_value

    def tilt_mapping(self, value, from_min=0.002479444699213061, from_max=0.5195337456759869, to_min=1, to_max=1.3):
        #   0.5811359661846451

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
        
        # print(offset_x, offset_y)

        # 1-2. 모든 goal_list에 있는 모든 좌표에 offset 적용
        self.global_path = Path()
        self.global_path.header.frame_id = 'map'

        for pos in range(len(self.goal_list)):
            self.goal_list[pos].pose.position.x += offset_x
            self.goal_list[pos].pose.position.y += offset_y

            pt = PoseStamped()
            pt.header.frame_id = 'map'
            pt.pose.position.x = self.goal_list[pos].pose.position.x
            pt.pose.position.y = self.goal_list[pos].pose.position.y
            pt.pose.orientation.z = self.goal_list[pos].pose.orientation.z
            pt.pose.orientation.w = self.goal_list[pos].pose.orientation.w
            
            self.global_path.poses.append(pt)

        self.path_pub.publish(self.global_path)

        # 2. 그리고 현재 내 위치를 기반으로 다시 시작 시퀀스를 찾아야 함 ==> 왜냐 self.lookahead 때문에 sequence 가 현재 내 위치 언저리를 가리키지 않은 상태이기 때문임
        # 일단 순차적으로 탐색하면서 내 방향이랑 비슷한 시퀀스 및 좌표를 가진 seq를 가져와야 하는 부분임
        # 기억에 mission45 가 적용되도록 해야 함
        # self.sequence = np.argmin(np.array([np.hypot(pos.pose.position.x - basis_x, pos.pose.position.y - basis_y) for pos in self.goal_list[0]]))
        # for seq in range(self.sequence, len(self.goal_list\\])):
        #     if self.dist(self.goal_list[0][seq].pose.position) < self.lookahead_distance:
        #         self.sequence = seq

        # # 위치 조정 후 발행 
        # self.republish_amcl.publish(init_pose)
        
if __name__ == "__main__":
    nc = Total()
    rospy.spin()
#! /usr/bin/env python3

import rospy
import pickle
from math import *

from geometry_msgs.msg import PoseWithCovarianceStamped,Twist
from lane_detection.msg import LaneInformation, PixelCoord
from nav_msgs.msg import Odometry # /odometry/filtered 토픽 수신 
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
import tf 


class NavigationClient:
    def __init__(self):
        self.now_pose = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)
        #rospy.Subscriber('/lane_information', LaneInformation, self.lane_callback)
        #ospy.Subscriber('/odometry/filtered',Odometry, self.Ld_callback)#현재 차량의 속도 받아오기 
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)# linear.x (진행방향 속도), angular.z(회전 각도)
        self.client=actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.goal_list = []

        self.vehicle_length = .26  # 차량 길이 설정
        #self.lookahead_distance = 0.8085 # Lookahead distance 설정
        # self.lookahead_distance = 0.3 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        #self.lookahead_distance = 0.598 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        self.lookahead_distance = 0.6 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =0.8 일때( 3km/h) 0.35
        self.angle_offset=0

        self.is_current_vel=False
        self.is_state=False
        self.target_vel=0
        self.current_vel=0.0

        self.L_curv=0
        self.R_curv=0


        
        with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'rb') as file:
            pt_list = pickle.load(file)

            for _pt in pt_list:
                pt = MoveBaseGoal()
                pt.target_pose.header.frame_id = 'map'
                pt.target_pose.pose.position.x = _pt.position.x
                pt.target_pose.pose.position.y = _pt.position.y
                pt.target_pose.pose.orientation.z = _pt.orientation.z
                pt.target_pose.pose.orientation.w = _pt.orientation.w
                
                self.goal_list.append(pt)

        self.goal_list.extend(self.goal_list[::-1]) #받아온 목표 좌표 모음

        # print(self.goal_list)
        self.sequence = 0 # 밫아온 좌표모음의 index
        self.start_time = rospy.Time.now()

        self.control_angle = Twist()

        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5

    # def lane_callback(self, msg: LaneInformation):
    #     #직선 = 곡률 반지름이 엄청 커지기에 곡률은 작다straight 보다 작으면 
    #     #곡선 
    #     self.L_curv = msg.left_gradient
    #     self.R_curv = msg.right_gradient
    #     L_point=msg.left_lane_points
    #     R_point=msg.right_lane_points

    #     reference_quat = self.goal_list[self.sequence+10].target_pose.pose.orientation
    #     reference_yaw = self.get_yaw_from_orientation(reference_quat)
    #     yaw = self.get_yaw_from_orientation(self.now_orientation) 
    #     # @차선 변경 조건 각도 (30도? 40도 추가해야함)
    #     if abs(yaw-reference_yaw)> 50:  #각도 차이가 어느정도 난다. 회전해야함 
    #         self.target_vel=0.8
    #         return
    #         #속도 줄이기 
    #     else: 
    #         self.control_angle.linear.x = 2  
    #             #8km/h 일 때 twist2.0 
    #         if abs(self.L_curv)>280 and abs(self.R_curv)>280: #양쪽 차선 모두 직선이거나, 왼쪽차선 없고 오른쪽 차선만 직선인 경우 속도 8km
    #             # @TODO: 그레디언트 값을 활용한 angle_offset 조정 
    #             self.angle_offset=-self.R_curv

    # def Ld_callback(self, data: Odometry):
    #     v = data.twist.twist.linear.x
        
    #     k_vel=0 #속도 게인값
    #     k_curv=0 #곡률 게인값
    #     k_offset=0 # Ld의 min~max값으로 맞추기

    #     if v < 0.9:
    #         self.lookahead_distance=(k_vel * v) + (k_curv /abs(self.R_curv)) + k_offset
        
    #     self.lookahead_distance=(k_vel * v) + (k_curv /abs(self.R_curv)) + k_offset
        
    #     # pub 발행, 생각해봐야 함
    #     # 주기가 다를 수 있으니 run 에서 방어 코드 작성해야 함
    #     self.run()
    
    
    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return euler[2]  # yaw

    def run(self):
        print(self.sequence)

        if not self.now_pose: return
        if self.dist(self.goal_list[self.sequence].target_pose.pose.position) < 0.39: 
            if self.sequence >= len(self.goal_list): 
                print('end~~')
                return
            self.sequence += 1

        dy = self.goal_list[self.sequence].target_pose.pose.position.y - self.now_pose.y
        dx = self.goal_list[self.sequence].target_pose.pose.position.x - self.now_pose.x
        
        # Pure Pursuit 알고리즘 적용
        angle_to_target = atan2(dy, dx)
        yaw = self.get_yaw_from_orientation(self.now_orientation)
        angle_difference = angle_to_target - yaw
        angle_difference = self.normalize_angle(angle_difference)

        # 조향각 계산

        gain = 1.
        steering_angle = gain * atan2(2.0 * self.vehicle_length * sin(angle_difference) / self.lookahead_distance, 1.0)  #arctan ( 2Lsin(a)/Ld) )        
        print('steer', steering_angle, 'angle_difference', angle_difference)


        speed=2.0
        # 조향각과 속도를 Twist 메시지로 전송
        self.control_angle = Twist()


        self.control_angle.angular.z = steering_angle
        self.control_angle.linear.x = speed
        self.vel_pub.publish(self.control_angle)          


    #조향각도를 -pi ~ +pi 범위로 정규화 (wrapping이라고 지칭)
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
    rospy.init_node('navigation_client')
    nc = NavigationClient()
    rate = rospy.Rate(10)

    # rospy.spin()
    while not rospy.is_shutdown():
        nc.run()
        rate.sleep()



if __name__=='__main__':
    main()

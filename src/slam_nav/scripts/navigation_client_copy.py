#! /usr/bin/env python3

import rospy
import pickle
from math import *

from geometry_msgs.msg import PoseWithCovarianceStamped,Twist

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
import tf 


class NavigationClient:
    def __init__(self):
        self.now_pose = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)# linear.x (진행방향 속도), angular.z(회전 각도)
        self.client=actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.goal_list = []

        self.vehicle_length = .26  # 차량 길이 설정
        #self.lookahead_distance = 0.8085 # Lookahead distance 설정
        # self.lookahead_distance = 0.3 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        #self.lookahead_distance = 0.598 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)
        self.lookahead_distance = 0.25 # 저속이니까 좀 작게 > 1/19 나쁘지 않았음 cmd vel =1 일때( 4km/h)


        
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

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return euler[2]  # yaw

    def run(self):
        print(self.sequence)
        
        if not self.now_pose: return
        if self.dist(self.goal_list[self.sequence].target_pose.pose.position) < 0.5:
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

        #delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
        # flag_angle = min(abs(steering_angle), 19.5)
        # gain = (flag_angle * pi) / (180 * 0.3403) #19.5 기준의 rad=0.3403

        # if gain >= 0.2:
        #     speed = 2.0 - gain * 1.2
        # else:        # 속도 설정
        #     speed = 2.0  # 고정 속도
        speed=0.8
        # 조향각과 속도를 Twist 메시지로 전송
        self.control_angle = Twist()
        self.control_angle.linear.x = speed
        self.control_angle.angular.z = steering_angle

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
    rospy.init_node('navigaion_client')
    nc = NavigationClient()
    rate = rospy.Rate(10)

    # rospy.spin()
    while not rospy.is_shutdown():
        nc.run()
        rate.sleep()


if __name__=='__main__':
    main()

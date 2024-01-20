#! /usr/bin/env python3

import rospy
import pickle
from math import atan2, sqrt

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
import tf

class NavigationClient:
    def __init__(self):
        self.now_pose = None
        self.now_orientation = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Load waypoints
        with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'rb') as file:
            self.goal_list = pickle.load(file)

        self.sequence = 0
        self.vehicle_length = .26  # 차량 길이 설정
        self.lookahead_distance = 1.0  # Lookahead distance 설정

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return euler[2]  # yaw

    def run(self):
        if not self.now_pose or not self.now_orientation: return

        if self.sequence < len(self.goal_list):
            target_point = self.goal_list[self.sequence]
            dx = target_point.position.x - self.now_pose.x
            dy = target_point.position.y - self.now_pose.y
            distance = sqrt(dx**2 + dy**2)

            if distance < self.lookahead_distance:
                self.sequence += 1  # 다음 웨이포인트로 이동
                if self.sequence >= len(self.goal_list):
                    self.stop()  # 목표지점 도달 시 정지
                    return

            # Pure Pursuit 알고리즘 적용
            angle_to_target = atan2(dy, dx)
            yaw = self.get_yaw_from_orientation(self.now_orientation)
            angle_difference = self.normalize_angle(angle_to_target - yaw)
            
            # 조향각 계산
            steering_angle = atan2(2.0 * self.vehicle_length * sin(angle_difference), self.lookahead_distance)  #arctan ( 2Lsin(a)/Ld) )

            # 속도 설정
            speed = 1.0  # 고정 속도

            # 조향각과 속도를 Twist 메시지로 전송
            twist = Twist()
            twist.linear.x = speed
            twist.angular.z = steering_angle
            self.vel_pub.publish(twist)

    def normalize_angle(self, angle):
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle

    def stop(self):
        twist = Twist()
        self.vel_pub.publish(twist)  # 속도를 0으로 설정하여 정지
        
def main():
    rospy.init_node('navigation_client')
    nc = NavigationClient()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        nc.run()
        rate.sleep()

if __name__ == '__main__':
    main()
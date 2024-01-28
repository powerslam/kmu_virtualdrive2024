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

        self.goal_list.extend(self.goal_list[::-1]) #받아온 목표 좌표 모음

        # print(self.goal_list)
        self.sequence = 0 # 밫아온 좌표모음의 index
        self.start_time = rospy.Time.now()

        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position
        self.now_orientation = msg.pose.pose.orientation

    def get_yaw_from_orientation(self, quat):
        euler = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        return euler[2]  # yaw

    def run(self):
        #print('hi1')
        if not self.now_pose: return
        #print('hi2')
        if self.client.get_state() != GoalStatus.ACTIVE:
            #print('hi3') 
            self.sequence = (self.sequence + 1) % len(self.goal_list)
            self.client.send_goal(self.goal_list[self.sequence])
            print(self.goal_list[self.sequence])

        elif self.dist(self.goal_list[self.sequence].target_pose.pose.position) < 0.1:
            self.stop()
        
    def stop(self):
        self.client.cancel_all_goals()
        twist = Twist()
        self.vel_pub.publish(twist)  # 속도를 0으로 설정하여 정지


def main():
    _ = NavigationClient()
    rospy.spin()


if __name__ == "__main__":
    main()

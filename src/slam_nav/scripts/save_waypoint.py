#! /usr/bin/env python3

import rospy
import pickle

from geometry_msgs.msg import PoseWithCovarianceStamped

class NavigationClient:
    def __init__(self):
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)
        self.pose_list = []
    
    def dist(self, now_pose: PoseWithCovarianceStamped):
        prev_pose = self.pose_list[-1].position
        now_pose = now_pose.pose.pose.position

        return ((prev_pose.x - now_pose.x) ** 2 + (prev_pose.y - now_pose.y) ** 2) ** 0.5

    def print(self):
        for pose in self.pose_list:
            print(pose.position)

    def callback(self, msg: PoseWithCovarianceStamped):
        self.print()

        if not self.pose_list:
            self.pose_list += [msg.pose.pose]

        elif self.dist(msg) > 0.5:
            self.pose_list += [msg.pose.pose]

def main():
    try:
        rospy.init_node('navigaion_client')
        nc = NavigationClient()
        rospy.spin()

        while not rospy.is_shutdown():
            nc.print()

        if rospy.is_shutdown():
            with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'wb') as file:
                pickle.dump(nc.pose_list, file)

    except rospy.ROSInterruptException:
        print('hello')


if __name__ == '__main__':
    main()
#! /usr/bin/env python3

import rospy
import pickle

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose

import numpy as np

# way point 작성 시 어떤 보간법이 필요함

class NavigationClient:
    def __init__(self):
        self.path_pub = rospy.Publisher('/save_path', Path, queue_size=1)
        self.rate = rospy.Rate(50)

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)
        self.pose_list = []
        self.path = Path()
        self.path.header.frame_id = 'map'
    
    def dist(self, now_pose: PoseWithCovarianceStamped):
        prev_pose = self.pose_list[-1].position
        now_pose = now_pose.pose.pose.position

        return ((prev_pose.x - now_pose.x) ** 2 + (prev_pose.y - now_pose.y) ** 2) ** 0.5

    def run(self):
        print(self.path)
        self.path_pub.publish(self.path)

    def callback(self, msg: PoseWithCovarianceStamped):
        # print(msg.pose.pose)

        if not self.pose_list:
            pose = PoseStamped()
            pose.pose = msg.pose.pose
            pose.header.frame_id = 'map'
            self.path.poses.append(pose)

            self.pose_list += [msg.pose.pose]

        elif self.dist(msg) > 0.05:
            pose = PoseStamped()
            pose.pose = msg.pose.pose
            pose.header.frame_id = 'map'
            self.path.poses.append(pose) 

            self.pose_list += [msg.pose.pose]
            # prev_pose = self.pose_list[-1].position
            # now_pose = msg.pose.pose.position

            # print(int(self.dist(msg) / 0.05) + 1)
            # _x = np.linspace(prev_pose.x, now_pose.x, int(self.dist(msg) / 0.05) + 1)
            # _y = np.linspace(prev_pose.y, now_pose.y, int(self.dist(msg) / 0.05) + 1)

            # for x, y in zip(_x[1:], _y[1:]):
            #     pose = Pose()
            #     pose.position.x = x
            #     pose.position.y = y
            #     pose.position.z = msg.pose.pose.position.z

            #     pose.orientation.w = msg.pose.pose.orientation.w
            #     pose.orientation.x = msg.pose.pose.orientation.x
            #     pose.orientation.y = msg.pose.pose.orientation.y
            #     pose.orientation.z = msg.pose.pose.orientation.z

            #     self.pose_list.append(pose)

def main():
    try:
        rospy.init_node('navigation_client')
        nc = NavigationClient()

        while not rospy.is_shutdown():
            nc.run()
            nc.rate.sleep()

        if rospy.is_shutdown():
            with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'wb') as file:
                pickle.dump(nc.pose_list, file)

    except rospy.ROSInterruptException:
        print('hello')


if __name__ == '__main__':
    main()
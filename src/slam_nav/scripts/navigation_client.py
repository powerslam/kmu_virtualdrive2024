#! /usr/bin/env python3

import rospy
import pickle

from geometry_msgs.msg import PoseWithCovarianceStamped

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib

class NavigationClient:
    def __init__(self):
        self.now_pose = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)

        self.client=actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.goal_list = []
        
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

        self.goal_list.extend(self.goal_list[::-1])

        # print(self.goal_list)
        self.sequence = 0
        self.start_time = rospy.Time.now()

        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position

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
            #print('hi4')
            self.stop()
        #print('hi5')    
        # if self.client.get_state() != GoalStatus.ACTIVE:
        #     self.start_time=rospy.Time.now()
        #     self.sequence=(self.sequence+1)%2
        #     self.client.send_goal(self.goal_list[self.sequence])
        # else:
        #     if (rospy.Time.now().to_sec() - self.start_time.to_sec()) > 30.0:
        #         self.stop()
        
    def stop(self):
        self.client.cancel_all_goals()
        
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

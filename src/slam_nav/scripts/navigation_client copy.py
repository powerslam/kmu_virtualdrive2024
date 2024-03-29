#! /usr/bin/env python3

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib

class NavigationClient:
    def __init__(self):
        self.now_pose = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback)

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.goal_list = []

        self.start = MoveBaseGoal()
        self.start.target_pose.header.frame_id='map'
        self.start.target_pose.pose.orientation.w=1.0

        self.goal_list.append(self.start)

        self.goal1=MoveBaseGoal()
        self.goal1.target_pose.header.frame_id='map'
        self.goal1.target_pose.pose.position.x=18.141439308224776
        self.goal1.target_pose.pose.position.y=-9.983781823848819
        self.goal1.target_pose.pose.orientation.z=-0.01639489184980084
        self.goal1.target_pose.pose.orientation.w=0.9998655947282281

        self.goal_list.append(self.goal1)

        self.goal2=MoveBaseGoal()
        self.goal2.target_pose.header.frame_id='map'
        self.goal2.target_pose.pose.position.x=34.149663085674376
        self.goal2.target_pose.pose.position.y=-9.983781823848819
        self.goal2.target_pose.pose.orientation.z=0.019714316866875418
        self.goal2.target_pose.pose.orientation.w=0.9998056539700465

        self.goal_list.append(self.goal2)

        self.sequence=0
        self.start_time=rospy.Time.now()

        self.dist = lambda pt: ((self.now_pose.x - pt.x) ** 2 + (self.now_pose.y - pt.y) ** 2) ** 0.5

    def callback(self, msg: PoseWithCovarianceStamped):
        self.now_pose = msg.pose.pose.position

    def run(self):
        if not self.now_pose: return
        if self.client.get_state() != GoalStatus.ACTIVE:
            self.sequence = (self.sequence + 1) % 3
            self.client.send_goal(self.goal_list[self.sequence])

        elif self.dist(self.goal_list[self.sequence].target_pose.pose.position) < 0.1:
            self.stop()

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

    while not rospy.is_shutdown():
        nc.run()
        rate.sleep()

if __name__=='__main__':
    main()
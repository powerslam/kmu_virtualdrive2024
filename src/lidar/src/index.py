import rospy
from sensor_msgs.msg import LaserScan
from math import *
import os
class Turtle_sub:
    def __init__(self):
        rospy.init_node("turtle_sub_node")
        rospy.Subscriber("/lidar2D",LaserScan,self.lidar_CB)
        self.scan_msg = LaserScan()
    
    def lidar_CB(self,msg):
        os.system("clear")
        self.scan_msg = msg
        degree_min = self.scan_msg.angle_min *180/pi
        degree_max = self.scan_msg.angle_max *180/pi
        degree_angle_increment = self.scan_msg.angle_increment * 180/pi
        print(self.scan_msg)
        print(degree_min)
        print(degree_max)
        print(degree_angle_increment)
        print(self.scan_msg.ranges)
        degrees = [degree_min +degree_angle_increment *index for index,value in enumerate(self.scan_msg.ranges)]
        print(degrees)
        for index, value in enumerate(self.scan_msg.ranges):
            if -30 < degrees[index]<30 and 0<value<3:
                print("obstacle")
                print(f"{index}:obstacle{degrees[index]}")
                


def main():
    try:
        turtle_sub = Turtle_sub()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()

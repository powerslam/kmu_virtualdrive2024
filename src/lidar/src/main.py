import rospy
from math import *

from sensor_msgs.msg import LaserScan
from lidar.msg import Obstacle, ObstacleArray

class ObstacleDetector:
    def __init__(self):
        rospy.init_node('lidar_sub')
        rospy.Subscriber('/lidar2D', LaserScan, self.callback)

        self.scan_msg = LaserScan()
        self.obstacle_infos_pub = rospy.Publisher('/obstacle_infos', ObstacleArray, queue_size = 10)

    def callback(self, msg: LaserScan):
        self.scan_msg = msg

        deg_min = self.scan_msg.angle_min * 180 / pi # -90
        deg_inc = self.scan_msg.angle_increment * 180 / pi # 90

        roi = self.scan_msg.ranges[180:] + self.scan_msg.ranges[:180]
        roi_degs = [deg_min + deg_inc * index for index in range(len(roi))]

        obst_size = 0
        prev_info = None
        obstacle_infos = ObstacleArray()
        for idx, dist in enumerate(roi):
            # 거리가 3.5 초과면 건너뜀
            if dist > 3.5:
                if obst_size > 10:
                    obstacle_info = Obstacle()

                    # 이거는 각도 값이 되어야 함
                    obstacle_info.start_deg = roi_degs[idx - obst_size]
                    obstacle_info.start_dist = roi[idx - obst_size]

                    obstacle_info.end_deg = roi_degs[idx - 1]
                    obstacle_info.end_dist = roi[idx - 1]

                    obstacle_infos.obstacle_infos.append(obstacle_info)

                    obst_size = 0
                    prev_info = None

            # 이전 정보가 None 이면 prev_info update
            elif not prev_info: 
                prev_info = dist
                obst_size += 1

            # 1도 사이의 값 간 거리가 2.0 보다 작으면 size += 1
            elif abs(dist - prev_info) < 2.0:
                prev_info = dist
                obst_size += 1

            # obst_size가 10이상이면
            else:
                obstacle_info = Obstacle()

                # 이거는 각도 값이 되어야 함
                obstacle_info.start_deg = roi_degs[idx - obst_size]
                obstacle_info.start_dist = roi[idx - obst_size]

                obstacle_info.end_deg = roi_degs[idx - 1]
                obstacle_info.end_dist = roi[idx - 1]
            
                obstacle_infos.obstacle_infos.append(obstacle_info)

                obst_size = 0
                prev_info = None
            
        self.obstacle_infos_pub.publish(obstacle_infos)

if __name__ == "__main__":
    try:
        obd = ObstacleDetector()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

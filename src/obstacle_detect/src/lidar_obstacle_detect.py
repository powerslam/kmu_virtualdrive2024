#!/usr/bin/env python3
#-*-coding:utf-8-*-

import rospy
import numpy as np

# lidar Subscriber 데이터
from sensor_msgs.msg import LaserScan

# publish 데이터
from obstacle_detect.msg import Rotary,RotaryArray                  # Rotary에 차량이 어디에 위치해 있는가
from obstacle_detect.msg import LidarObstacleInfo, LidarObstacleInfoArray   # 라이다에서 취득한 장애물 정보

from bisect import bisect_left as lower_bound

class LidarObstacle:
    def __init__(self):
        rospy.init_node("lidar_obstacle")
        
        self.scan_msg = LaserScan()
        rospy.Subscriber("/lidar2D", LaserScan, self.callback)

        self.obstacle_pub = rospy.Publisher("/lidar_obstacle_information", LidarObstacleInfoArray, queue_size=10)
        self.rotary_pub = rospy.Publisher("/rotary_info", RotaryArray, queue_size=10)
        self.degrees = range(-180, 180)

    def callback(self, msg: LaserScan):
        self.scan_msg = msg
        is_searching_obstacle = False # 현재 장애물 인덱싱 중인가?

        obstacle_prev_deg = 0 
        obstacle_start_deg = 0 # 임의의 장애물 첫 시작 인덱스

        obst_size = 0
        obstacle_arr = LidarObstacleInfoArray()
        ranges = self.scan_msg.ranges[:180][::-1] + self.scan_msg.ranges[180:][::-1]
        
        for index, value in enumerate(ranges):
            if self.degrees[index] < -115 or self.degrees[index] > 115: continue
            # print(ranges[index], end=' ')

            # 3. 이내로 들어오는 경우
            if 0 <= value <= 2.5:
                if not is_searching_obstacle:  # 처음 장애물 인덱스 판단을 실시할 때                    
                    # 시작했으니 1로 변경
                    is_searching_obstacle = True
                    obst_size = 0

                    obstacle_prev_deg = self.degrees[index]
                    obstacle_prev_dist = ranges[index]
                    obstacle_start_deg = self.degrees[index]
                    
                elif abs(ranges[index] - obstacle_prev_dist) < 1.:
                    obst_size += 1
                    obstacle_prev_deg = self.degrees[index]
                    obstacle_prev_dist = ranges[index]
                
                # 장애물 찾는 중
                elif abs(self.degrees[index] - obstacle_prev_deg) < 8:
                    obst_size += 1
                    obstacle_prev_deg = self.degrees[index]
                    obstacle_prev_dist = ranges[index]
                
            
            elif is_searching_obstacle:
                # 장애물의 크기가 너무 작으면(3보다 작은 경우) skip
                if obst_size < 2 or obst_size > 40:
                    obst_size = 0
                    is_searching_obstacle = False

                else:
                    middle_index = (obstacle_start_deg + obstacle_prev_deg) / 2
                    middle_value = ranges[min(lower_bound(self.degrees, int(middle_index)), 359)]

                    x = middle_value * np.sin(middle_index * np.pi / 180)
                    y = middle_value * np.cos(middle_index * np.pi / 180)
                    
                    obstacle_arr.obstacle_infos.append(LidarObstacleInfo(obst_x = x, obst_y = y))

                    is_searching_obstacle = False

        # print()
        self.obstacle_pub.publish(obstacle_arr)

        rotary_infos = RotaryArray()

        for info in obstacle_arr.obstacle_infos:
            rotary = Rotary()
            x, y = info.obst_x, info.obst_y

            rotary.dis = np.hypot(x, y)
            rotary.orientation = ord('l') if x < 0 else ord('r')

            rotary_infos.moving_cars.append(rotary)

        if not len(obstacle_arr.obstacle_infos):
            rotary_infos.moving_cars.append(Rotary(dis=-10000, orientation=ord('n')))

        self.rotary_pub.publish(rotary_infos)

def main():
    try:
        _ = LidarObstacle()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()

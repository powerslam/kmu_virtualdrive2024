import rospy
from sensor_msgs.msg import LaserScan
from math import *
import os

from bisect import bisect_left as lower_bound
from lidar.msg import Rotary
from std_msgs.msg import Float32MultiArray

class Rotary_sub:
    def __init__(self):
        rospy.init_node("Rotary_sub_node")
        rospy.Subscriber("/lidar2D",LaserScan,self.lidar_CB)
        self.scan_msg = LaserScan()
        self.rotary_pub = rospy.Publisher("/rotary_info", Rotary, queue_size=10)
    
    def lidar_CB(self, msg: LaserScan):
        index_two =0 
        obstacle_index = 0 # 장애물 개수
        obstacle_flag = 0 # 현재 장애물 인덱싱 중인가?
        prev_flag = 0 
        start_flag = 0 # 임의의 장애물 첫 시작 인덱스
        finish_flag = 0 # 임의의 장애물 마지막 인덱스
        self.scan_msg = msg
        degree_min = self.scan_msg.angle_min * 180 / pi
        degree_max = self.scan_msg.angle_max * 180 / pi
        degree_angle_increment = self.scan_msg.angle_increment * 180/pi
        middle_index = 0
        obstacle_middle= []
        value_middle=[]
        obstacle_start= []
        value_start=[]
        obstacle_finish = []
        value_finish = []
        final_value = 0 # 원점으로 부터 떨어진 거리
        final_angle = 0 
        value_final=[]
        degrees = [degree_min + degree_angle_increment * index for index in range(len(self.scan_msg.ranges))]

        ranges = self.scan_msg.ranges[:180][::-1] + self.scan_msg.ranges[180:][::-1]
        
        for index, value in enumerate(ranges):
            if (-90 < degrees[index] < 90 ) and 0 <= value < 2:
                if obstacle_flag == 0:  # 장애물 인덱스 판단을 실시할 때                    
                    index += 1
                    obstacle_flag = 1
                    prev_flag = degrees[index]
                    start_flag = degrees[index]
                    start_value = value
                    prev_value = value
                    obstacle_start.append(start_flag)
                    value_start.append(start_value)
                 
                  
                elif obstacle_flag ==1 and abs(degrees[index] - prev_flag) < 8:
                    obstacle_flag =1
                    index_two =1
                    
                    prev_flag = degrees[index]
                    prev_value = value

                elif obstacle_flag == 1 and abs(degrees[index] - prev_flag) >= 8:  # 이제는 장애물 한 턴이 끝났다고 생각해야 함
                    obstacle_flag = 0
                    finish_flag = prev_flag
                    finish_value = prev_value

                    obstacle_index += 1
                    
                    
                    obstacle_finish.append(finish_flag)
                    value_finish.append(finish_value)

                    middle_index = (start_flag + finish_flag) / 2.0
                    middle_value = ranges[min(lower_bound(degrees, int(middle_index)), 359)]

                    value_middle.append(middle_value)
                    obstacle_middle.append(middle_index)
                    
                    value_final.append(middle_value * sin(obstacle_middle[-1] * pi / 180))

                    

                    start_flag =0
                    finish_flag =0 
                  
            else :
                if(obstacle_flag==1 and index_two==1):
                    obstacle_index += 1
                    
                    
                    finish_flag = prev_flag
                    finish_value= prev_value
                    
                    obstacle_finish.append(finish_flag)
                    value_finish.append(finish_value)

                    middle_index = (start_flag + finish_flag) / 2.0
                    middle_value = ranges[min(lower_bound(degrees, int(middle_index)), 359)]

                    value_middle.append(middle_value)
                    obstacle_middle.append(middle_index)
                    value_final.append(middle_value * sin(obstacle_middle[-1] * pi / 180))
                    
                    start_flag = 0
                    finish_flag = 0
                    obstacle_flag = 0
                    index_two = 0

        # 무조건 로터리 정지선 앞에서 한다는 조건
        rotary_data = Rotary()

        if obstacle_index == 1:
            rotary_data.dis = value_middle[-1]
            # l n r
            rotary_data.orientation = ord('l') if value_final[-1] < 0 else ord('r')
        else:
            rotary_data.dis = -10000
            rotary_data.orientation = ord('n')
            
        self.rotary_pub.publish(rotary_data)

def main():
    try:
        _ = Rotary_sub()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()

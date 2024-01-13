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
        #os.system("clear") 
        cnt = 0
        index_avg = 0
        index_sum = 0 
        obstacle_index = 0 # 장애물 개수
        obstacle_flag = 0 # 현재 장애물 인덱싱 중인가?
        prev_flag = 0 
        start_flag = 0 # 임의의 장애물 첫 시작 인덱스
        finish_flag = 0 # 임의의 장애물 마지막 인덱스
        self.scan_msg = msg
        degree_min = self.scan_msg.angle_min *180/pi
        degree_max = self.scan_msg.angle_max *180/pi
        degree_angle_increment = self.scan_msg.angle_increment * 180/pi
        #print(self.scan_msg)
        #print(degree_min)
        #print(degree_max)
        #print(degree_angle_increment)
        #print(self.scan_msg.ranges)
        degrees = [degree_min +degree_angle_increment *index for index,value in enumerate(self.scan_msg.ranges)]
        #print(degrees)
        
        # 장애물 인덱싱도 해야 한다. 첫번째 장애물 뭉텅이의 처음과 끝, 두번째 장애물 뭉텅이의 처음과 끝
        for index, value in enumerate(self.scan_msg.ranges):
            if -90 < degrees[index] < 90 and 0 <= value < 5:
                #print(f"obstacle: {degrees[index]}")
                if obstacle_flag == 0:  # 장애물 인덱스 판단을 실시할 때
                    index_sum += degrees[index]
                    index += 1
                    obstacle_flag = 1
                    obstacle_index = 1
                    prev_flag = degrees[index]
                    start_flag = degrees[index]
                    #print(f"start: {start_flag}")
                elif obstacle_flag ==1 and abs(degrees[index] - prev_flag) < 2.0:
                    obstacle_flag =1
                    index_sum += degrees[index]
                    cnt += 1
                    prev_flag = degrees[index]

                elif obstacle_flag == 1 and abs(degrees[index] - prev_flag) >= 2.0:  # 이제는 장애물 한 턴이 끝났다고 생각해야 함
                    obstacle_flag = 0
                    finish_flag = prev_flag
                    index_avg = index_sum / cnt
                    print(f"avg : {index_avg}")
                    index_sum = 0
                    cnt = 0
                    #print(f"finish : {finish_flag}")
                    obstacle_index += 1 # 뭔가 장애물 인덱스 값이 축적되는 기분이 든다
                
        print(f"장애물 개수 : {obstacle_index}")        
        print("here")

                
def main():
    try:
        turtle_sub = Turtle_sub()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()

# 1/15 정면 각도 수정 
# 3.5 ~ 1.3m 이내의 장애물 판단
# 장애물 중점 위치 (아마 자동차 중심으로 각도로 표현해야 함.)
# 내가 만든 정보 펍 하는 코드까지 완성하고 집 가야함. 
# 벽에 가까이 있을 경우 장애물이 벽과 구별
import rospy
from sensor_msgs.msg import LaserScan
from math import *
import os
from std_msgs.msg import Float32MultiArray
class Turtle_sub:
    def __init__(self):
        rospy.init_node("turtle_sub_node")
        rospy.Subscriber("/lidar2D",LaserScan,self.lidar_CB)
        self.scan_msg = LaserScan()
        self.obstacle_pub = rospy.Publisher("/obstacle_info", Float32MultiArray, queue_size=10)
    def lidar_CB(self,msg):
        cnt = 1
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
        middle_index = 0
        obstacle_middle= []

        #print(self.scan_msg)
        #print(degree_min)
        #print(degree_max)
        #print(degree_angle_increment)
        #print(self.scan_msg.ranges)
        degrees = [degree_min +degree_angle_increment *index for index,value in enumerate(self.scan_msg.ranges)]
        #print(degrees)
            
        for index, value in enumerate(self.scan_msg.ranges):
            
            if -180 < degrees[index] < -170 :
               print(f"각도 : {degrees[index]} 정면거리 : {value}")
            
            if (-180 < degrees[index] < -145 or 145< degrees[index]< 180) and 0 <= value < 3.5:
                if obstacle_flag == 0:  # 장애물 인덱스 판단을 실시할 때
                    index_sum += degrees[index]
                    index += 1
                    obstacle_flag = 1
                    prev_flag = degrees[index]
                    start_flag = degrees[index]
                    print(f"1: value:  {value} , index : {degrees[index]}")
                    start_flag = degrees[index]
                    #print(f"start: {start_flag}")
                elif obstacle_flag ==1 and abs(degrees[index] - prev_flag) < 2.0:
                    obstacle_flag =1
                    index_two =1
                    index_sum += degrees[index]
                    cnt += 1
                    prev_flag = degrees[index]
                    print(f"2: value:  {value} , index : {degrees[index]}")

                elif obstacle_flag == 1 and abs(degrees[index] - prev_flag) >= 2.0:  # 이제는 장애물 한 턴이 끝났다고 생각해야 함
                    obstacle_flag = 0
                    finish_flag = prev_flag
                    index_sum = 0
                    obstacle_index += 1
    
                    middle_index = (start_flag + finish_flag)/2.0
                    obstacle_middle.insert(obstacle_index,middle_index)
                    print(f"insert : {obstacle_index}")

                    start_flag =0
                    finish_flag =0 
                    print(f"3: value:  {value} , index : {degrees[index]}")
                    #print(f"middle_index : {middle_index}")
            else :
                if(obstacle_flag==1 and index_two==1):
                    obstacle_index += 1
            
                    finish_flag = prev_flag
                    middle_index = (start_flag + finish_flag)/2.0
                    obstacle_middle.insert(obstacle_index,middle_index)
                    #print(f"middle_index : {middle_index}")
                    start_flag =0
                    finish_flag =0
                    obstacle_flag = 0
                    index_two = 0
                    print(f"insert : {obstacle_index}")
        print(f"장애물 개수 : {obstacle_index}")  
        for index in range(1,obstacle_index+1):
            print(f"middle index {index} : {obstacle_middle[index-1]}")
        # 장애물 개수만큼 반복문 돌면서 배열안에 저장된 장애물의 중심 각도를 같이 프린트 해야함.       
        print("here")

def main():
    try:
        turtle_sub = Turtle_sub()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()
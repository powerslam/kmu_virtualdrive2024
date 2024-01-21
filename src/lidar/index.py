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
        
        index_two =0 
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
        value_middle=[]
        obstacle_start= []
        value_start=[]
        obstacle_finish = []
        value_finish = []
        final_value = 0 # 원점으로 부터 떨어진 거리
        final_angle = 0 
        value_final=[]
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
            
            if (-180 < degrees[index] < -135 or 135< degrees[index]< 180) and 0 <= value < 3.5:
                if obstacle_flag == 0:  # 장애물 인덱스 판단을 실시할 때
                    
                    index += 1
                    obstacle_flag = 1
                    prev_flag = degrees[index]
                    start_flag = degrees[index]
                    start_value = value
                    prev_value = value
                    obstacle_start.append(start_flag)
                    value_start.append(start_value)
                    print(f"1: value:  {value} , index : {degrees[index]}")
                    #print(f"start: {start_flag}")
                elif obstacle_flag ==1 and abs(degrees[index] - prev_flag) < 8:
                    obstacle_flag =1
                    index_two =1
                    
                    prev_flag = degrees[index]
                    prev_value = value

                    print(f"2: value:  {value} , index : {degrees[index]}")

                elif obstacle_flag == 1 and abs(degrees[index] - prev_flag) >= 8:  # 이제는 장애물 한 턴이 끝났다고 생각해야 함
                    obstacle_flag = 0
                    finish_flag = prev_flag
                    finish_value = prev_value

                    obstacle_index += 1
                    print("obstacle_append")
                    
                    obstacle_finish.append(finish_flag)
                    value_finish.append(finish_value)
                    middle_index = (start_flag + finish_flag)/2.0
                    middle_value = (finish_value+start_value)/2.0
                    value_middle.append(middle_value)
                    if(middle_index<180):
                        final_angle = 90-(180-middle_index) #중심선으로 부터 떨어진 각도로 측정
                    else:
                        final_angle = 90-(-180-middle_index) #중심선으로 부터 떨어진 각도로 측정
                    final_value = middle_value * cos(final_angle) 
                    value_final.append(final_value)
                    obstacle_middle.append(middle_index)
                    print(f"insert : {obstacle_index}")

                    start_flag =0
                    finish_flag =0 
                    print(f"3: value:  {value} , index : {degrees[index]}")
                    #print(f"middle_index : {middle_index}")
            else :
                if(obstacle_flag==1 and index_two==1):
                    obstacle_index += 1
                    print("obstacle_append")
                    finish_flag = prev_flag
                    finish_value= prev_value
                    obstacle_finish.append(finish_flag)
                    value_finish.append(finish_value)
                    middle_index = (start_flag + finish_flag)/2.0
                    middle_value = (finish_value+start_value)/2.0
                    value_middle.append(middle_value)
                    obstacle_middle.append(middle_index)
                    if(middle_index<180):
                        final_angle = 90-(180-middle_index) #중심선으로 부터 떨어진 각도로 측정
                    else:
                        final_angle = 90-(-180-middle_index) #중심선으로 부터 떨어진 각도로 측정
                    final_value = middle_value * cos(final_angle) 
                    value_final.append(final_value)
                    #print(f"middle_index : {middle_index}")
                    start_flag =0
                    finish_flag =0
                    obstacle_flag = 0
                    index_two = 0
                    print(f"insert : {obstacle_index}")
        obstacle_data = Float32MultiArray(data=[])
        for index in range(1,obstacle_index+1):
            obstacle_data.data.append(obstacle_index)
            obstacle_data.data.append(obstacle_start[index])
            obstacle_data.data.append(obstacle_middle[index-1])
            obstacle_data.data.append(obstacle_finish[index-1])   
            obstacle_data.data.append(value_start[index])
            obstacle_data.data.append(value_middle[index-1])         
            obstacle_data.data.append(value_finish[index-1])         
            
        self.obstacle_pub.publish(obstacle_data)
        
        print(f"장애물 개수 : {obstacle_index}")  
        for index in range(1,obstacle_index+1):
            print(f"start index {index} : {obstacle_start[index-1]} start_value : {value_start[index-1]}")
            print(f"middle index {index} : {obstacle_middle[index-1]} middle_value : {value_middle[index-1]}")
            print(f"finish index {index} : {obstacle_finish[index-1]} finish_value : {value_finish[index-1]}")
            print(f"final value {index} :  final_value : {value_final[index-1]}")
            
        print("here")

def main():
    try:
        turtle_sub = Turtle_sub()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__": 
    main()

#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
import rospkg 
import numpy as np
from math import cos, sin, pi,sqrt,pow,atan2

from geometry_msgs.msg import Point 
from nav_msgs.msg import Odometry, Path 
from morai_msgs.msg import CtrlCmd #제어 메시지 이거 써도 되나? velocity/steering입력에 관한 애들 
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#gps좌표계를 안쓰기에 좌표계 변환은 필요없을 듯 ?
class pure_pursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit',anonymous=True)
        rospy.Subscriber('local_path',Path,self.path_callback) #결국여기서 말하는 local path라는 게 pure purusit
        #local_path라는 토픽이이 들어오면 path_callback함수를 수행해라. 이때 토픽의 데이터형식은 Path형식이다

        rospy.Subscriber('odom',Odometry,self.odom_callback) #gps,imu데이터 변환해서 odom형식으로 publish한다는데 gps는 우리가 안씀 
        #odom토픽에서 Odometry데이터 수신,
        self.ctrl_cmd_pub=rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)

        #우리가 넣을 제어 메시지 형식에 맞게 수정(아마 amcl에서 쓰는 amcl pose의 데이터 타입으로 수정할듯 )
        self.ctrl_cmd_msg=CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType=2 #longCmdType: 제어방식을 결정하는 인덱스, Acceleration 제어6
        #2는 velocity, 제어, velocity/steering만 사용 

        self.is_path=False
        self.is_odom=False
        
        self.forward_point=Point()
        self.current_position=Point()
        self.is_look_forward_point=False
        self.vehicle_length=None #차량 휠베이스길이
        self.lfd=None #lookahead distance는 차량속도에 맞게 수정 
        
        rate=rospy.Rate(15) #15hz
        while not rospy.is_shutdown():
            if self.is_path==True and self.is_odom==True:

                vehicle_position=self.current_position
                self.is_look_forward_point=False
                
                translation=[vehicle_position.x, vehicle_position.y]

                t=np.array([
                    [cos(self.vehicle_yaw),-sin(self.vehicle_yaw),translation[0]],
                    [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                    [0,                       0,                  1           ]])
                

                det_t=np.array([
                    [t[0][0],t[1][0],-(t[0][0]*translation[0]+t[1][0]*translation[1])],
                    [t[0][1],t[1][1],-(t[0][1]*translation[0]+t[1][1]*translation[1])],
                    [0       ,0       ,1             ]])
                
                for num, i in enumerate(self.path.poses): #path.poses 배열의 원소와 인덱스 반환
                    path_point=i.pose.position
                    
                    global_path_point=[path_point.x,path_point.y,1]#전역경로상 point
                    local_path_point=det_t.dot(global_path_point)#오일러좌표계 쿼터니안으로 변환(현재 차량 위치)
                    if local_path_point[0]>0:
                        dis=sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2)) #쿼터니안 (x,y,z,w) 순 인덱스임 
                        if dis>=self.lfd: # 이 조건이 어떤걸 의미하지? 
                            self.forward_point=path_point
                            self.is_look_forward_point=True
                            break
                    
                theta=atan2(local_path_point[1],local_path_point[0])
                if self.is_look_forward_point:
                    self.ctrl_cmd_msg.steering=atan2((2*self.vehicle_length*sin(theta)),self.lfd)
                    self.ctrl_cmd_msg.velocity=20.0
                    print(self.ctrl_cmd_msg.steering)
          
                else:
                    print("no found forward point")
                    self.ctrl_cmd_msg.steering=0.0
                    self.ctrl_cmd_msg.velocity=0.0

                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
            
            rate.sleep()
                

        def path_callback(self,msg):
            self.is_path=True
            self.path=msg
        def odom_callback(self,msg):
            self.is_odom=True
            odom_quarternion=(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
            _,_,self.vehicle_yaw=euler_from_quaternion(odom_quarternion)
            self.current_position.x=msg.pose.pose.position.x
            self.current_position.y=msg.pose.pose.position.y
        
if __name__=="__main__":
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass

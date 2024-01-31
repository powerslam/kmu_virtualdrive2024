import rospy, math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

class Cmd2Ack:
    def __init__(self):
        rospy.init_node('cmd_vel_to_ackerman', anonymous=True)

        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback, queue_size=10)
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
        self.wheel_base = 0.26

        self.msg = AckermannDriveStamped()
        self.msg.header.frame_id = 'odom'

    def convert_trans_rot_vel_to_steering_angle(self, v, omega, wheelbase):
        if omega == 0 or v == 0:
            return 0
        
        radius = v / omega
        return math.atan(wheelbase / radius)

    def cmd_callback(self, data: Twist):
        v = data.linear.x
        steering = self.convert_trans_rot_vel_to_steering_angle(v, data.angular.z, self.wheel_base)

        self.msg.header.stamp = rospy.Time.now()
        self.msg.drive.steering_angle = steering
        self.msg.drive.steering_angle_velocity
        self.msg.drive.speed = v
        if v <= 0.4 :
            self.msg.drive.acceleration = 10000000

        self.ack_pub.publish(self.msg)

if __name__ == '__main__':
    try:
        cmd_2_ack = Cmd2Ack()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass

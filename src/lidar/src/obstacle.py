import numpy as np
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud2
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped

def dbscan_clustering(pointcloud_msg,publish_topic):
    pc_data = pc2.read_points(pointcloud_msg,field_names=("x","y","z"),skip_nans=True)
    data= np.array(list(pc_data))
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    labels = dbscan.fit_predict(data)
    for label in set(labels) :
        if label == -1:
            continue
        cluster_points = data[labels==label]
        cluster_center = np.mean(cluster_points,axis=0)

        if np.linalg.norm(cluster_center[:2])<10:
            publish_cluster_center(cluster_center,publish_topic)

def publish_cluster_center(cluster_center,publish_topic):
    center_msg = PointStamped()
    center_msg.header.stamp =rospy.Time.now()
    center_msg.point.x = cluster_center[0]
    center_msg.point.y = cluster_center[1]
    center_msg.point.z = cluster_center[2]

    publisher.publish(center_msg)

def main():
    rospy.init_node('obstacle_detection_node')
    lidar_topic = '/lidar2D'
    rospy.Subscriber(lidar_topic,PointCloud2,dbscan_clustering,callback_args='/obstacle_center')
    global publisher
    publisher = rospy.Publisher('obstacle_center',PointStamped,queue_size=10)
    
    rospy.spin()

if __name__ == "__main__":
    main()
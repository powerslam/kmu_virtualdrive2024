import pickle
import numpy as np

with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'rb') as file:
    pose_list = pickle.load(file)

    amcl_x1 = 36.75049579024794
    amcl_y1 = -7.902592948142209
    amcl_x2 = 34.2791376099585
    amcl_y2 = -9.69619024089615

    for idx, pose in enumerate(pose_list):
        if pose.position.x == amcl_x1 and pose.position.y == amcl_y1:
            print('start idx', idx)

        if pose.position.x == amcl_x2 and pose.position.y == amcl_y2:
            print('end idx', idx)

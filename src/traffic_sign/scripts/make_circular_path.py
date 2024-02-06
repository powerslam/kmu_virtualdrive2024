import pickle
import numpy as np

with open("/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl", "rb") as file:
    waypoint = pickle.load(file)

# for idx, point in enumerate(waypoint):
#     if (37.149 < point.position.x < 37.150):
#         print('1: ', idx)
    
#     if (34.465 < point.position.x < 34.466):
#         print('2: ', idx)

# print(waypoint[365:447])

x = np.linspace(34.4659, 36.83304, 67)
print(x)

y = []
for _x in x:
    print(2.36715 ** 2 - (_x - 34.4659) ** 2)
    y += [np.sqrt(2.36715 ** 2 - (_x - 34.4659) ** 2) - 7.6513]

from geometry_msgs.msg import Pose
print(y)

new_pose_list = []
for _ in range(len(x)):
    pose = Pose()

    pose.position.x = 

    new_pose_list += [pose]

print(waypoint[365:447])

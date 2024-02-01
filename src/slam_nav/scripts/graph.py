import numpy as np
import pickle
import matplotlib.pyplot as plt

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
import tf

def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)

# amcl 좌표계
amcl_x1 = 36.75049579024794
amcl_y1 = -7.902592948142209
amcl_x2 = 34.2791376099585
amcl_y2 = -9.69619024089615

# 데카르트 좌표계
deca_x1 = -amcl_y1
deca_y1 = amcl_x1
deca_x2 = -amcl_y2
deca_y2 = amcl_x2

print('데카르트 좌표')
print('A', deca_x1, deca_y1, 'B', deca_x2, deca_y2)

deca_O = np.array([deca_x1, deca_y2])
print('원점', deca_O)

# plt.scatter([0, deca_O[0]], [0, deca_O[1]], c='r')

x_diff = (deca_x2 - deca_O[0]) ** 2
y_diff = (deca_y1 - deca_O[1]) ** 2

# 데카르트 좌표계 원의 방정식
x = np.linspace(deca_x1, deca_x2, 10000)

final_x = x - x[-1]
final_y = []
for _x in final_x:
    y_value = np.sqrt(y_diff * (1 - (_x - final_x[0]) ** 2 / x_diff))
    final_y.append(y_value)

plt.scatter(final_x, final_y)

res = [[final_x[0], final_y[0]]]
for _x, _y in zip(final_x[1:], final_y[1:]):
    if np.hypot(_x - res[-1][0], _y - res[-1][1]) < 0.05:
        continue

    else:
        res.append([_x, _y])
        prev = res[-1]

res = np.array(res)
# for i in range(1, len(res)):
#     print(np.hypot(res[i][0] - res[i - 1][0], res[i][1] - res[i - 1][1]))

print(len(res))
plt.scatter(res[:, 0], res[:, 1], c='r')  # Corrected the order of x and y for the red points

res[:, 0] += x[-1]
res[:, 1] += deca_O[1]

print(res)

with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'rb') as file:
    pose_list = pickle.load(file)

    amcl_x1 = 36.048006190045676
    amcl_y1 = -7.792827774590465
    amcl_x2 = 33.892257404326536
    amcl_y2 = -9.736606185018964

    for idx, pose in enumerate(pose_list):
        if pose.position.x == amcl_x1 and pose.position.y == amcl_y1:
            print('start idx', idx)

        if pose.position.x == amcl_x2 and pose.position.y == amcl_y2:
            print('end idx', idx)

    pose_res = []
    for _x, _y in res[::-1]:
        pose = Pose()
        pose.position.x = _y
        pose.position.y = -_x
        pose.position.z = 0.

        yaw = -(y_diff/x_diff) * ((_x - res[0][0])/np.sqrt(x_diff - (_x - res[0][0]) ** 2))
        quat = tf.transformations.quaternion_from_euler(0., 0., yaw)
        print(quat)

        pose.orientation.x = 0.
        pose.orientation.y = 0.
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        pose_res += [pose]

    for pose in pose_list[200:400]:
        print('pose', end=' ')
        print(pose.position)
        print('orientation', end=' ')
        print(pose.orientation)

    # pose_list[286:344] = pose_res
    # print(pose_list[285:400])

    # print(pose_list[290:360])

# with open('/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/waypoint.pkl', 'wb') as file:
#     pickle.dump(pose_list, file)


# rot = np.ones((len(res), 3))
# rot[:, 0] = res[:, 0]
# rot[:, 1] = res[:, 1]

# rot1 = rotation_matrix(np.pi / 2) @ rot.T
# plt.scatter(rot1.T[:, 0], rot1.T[:, 1], c='g')
# print(rot1.T)

# rot2 = rotation_matrix(np.pi) @ rot.T
# plt.scatter(rot2.T[:, 0], rot2.T[:, 1], c='r')
# print(rot2.T)

# rot3 = rotation_matrix(3 * np.pi / 2) @ rot.T
# plt.scatter(rot3.T[:, 0], rot3.T[:, 1], c='g')
# print(rot3.T)

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')  # Corrected axis labels
plt.show()
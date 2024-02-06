import pickle

mission45_goal = []
with open("/home/foscar/kmu_virtualdrive2024/src/slam_nav/scripts/mission45.pkl", "rb") as file:
    pt_list = pickle.load(file)
    print(len(pt_list))
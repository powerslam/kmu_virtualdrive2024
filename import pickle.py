import rosbag

bag = rosbag.Bag('2024-02-04-08-01-17.bag')

for topic, msg, t in bag.read_messages():
    print(f"Topic: {topic}, Message: {msg}, Timestamp: {t}")

bag.close()

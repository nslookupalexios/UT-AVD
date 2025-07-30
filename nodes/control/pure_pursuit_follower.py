#!/usr/bin/env python3
import rospy

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped

class PurePursuitFollower:
    def __init__(self):

        # Parameters

        # Publishers

        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        print(msg.header.seq)

    def current_pose_callback(self, msg):
        print(msg.header.seq)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()
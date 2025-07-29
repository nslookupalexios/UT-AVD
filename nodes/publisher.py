#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
message = str(rospy.get_param('~message', 'Hello World!'))
rate = rospy.Rate(int(rospy.get_param('~hz', 2)))
pub = rospy.Publisher('/message', String, queue_size=10)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()
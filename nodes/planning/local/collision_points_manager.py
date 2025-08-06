#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")

        # variables
        self.detected_objects = None
        self.goal_waypoint = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def global_path_callback(self, msg):
        if len(msg.waypoints) == 0:
            rospy.logwarn_throttle(5, "%s - Received global path with no waypoints!", rospy.get_name())
            return

        # extract the last waypoint as the goal
        with self.lock:
            self.goal_waypoint = msg.waypoints[-1]
            rospy.loginfo_throttle(5, "%s - Updated goal waypoint at (%.2f, %.2f)", rospy.get_name(),
                                self.goal_waypoint.position.x, self.goal_waypoint.position.y)

    
    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)


        if len(msg.waypoints) > 0:

            local_path_linestring = shapely.LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
            local_path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
            shapely.prepare(local_path_buffer)

            if len(detected_objects) > 0:

                for obj in detected_objects:
                    convex_hull_array = np.array(obj.convex_hull).reshape(-1, 3)
                    convex_hull_2d = convex_hull_array[:, :2]
    
                    object_polygon = shapely.Polygon(convex_hull_2d)

                
                    if not object_polygon.intersects(local_path_buffer):
                        continue

                    intersection_geom = object_polygon.intersection(local_path_buffer)
                    intersection_coords = shapely.get_coordinates(intersection_geom)

                    object_speed = np.linalg.norm([
                        obj.velocity.x,
                        obj.velocity.y,
                        obj.velocity.z
                    ])

                    for x, y in intersection_coords:
                        collision_points = np.append(collision_points, np.array([(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                                                                                                    self.braking_safety_distance_obstacle, np.inf, 3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))

            # goal waypoint processing
            if self.goal_waypoint is not None:
                goal_point = shapely.Point(self.goal_waypoint.position.x, self.goal_waypoint.position.y)
                goal_buffer = goal_point.buffer(0.5) # 0.5 expands the goal point slightly (into a small circular area) and gives tolerance to account for small path/goal mismatches
                if goal_buffer.intersects(local_path_buffer):
                    collision_points = np.append(collision_points, np.array([(goal_point.x, goal_point.y, self.goal_waypoint.position.z,
                                                                            0.0, 0.0, 0.0,  # no velocity because its a static point
                                                                            self.braking_safety_distance_goal, np.inf,
                                                                            1)],  # category 1 = goal
                                                                            dtype=DTYPE))

            
            pointcloud_msg = msgify(PointCloud2, collision_points)
            pointcloud_msg.header = msg.header
            self.local_path_collision_pub.publish(pointcloud_msg)
            return

        empty_message = PointCloud2()
        empty_message.header = msg.header
        self.local_path_collision_pub.publish(empty_message)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()
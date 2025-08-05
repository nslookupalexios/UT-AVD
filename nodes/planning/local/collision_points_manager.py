#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from shapely.geometry import LineString, Polygon
from shapely import get_coordinates, intersection, prepare
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
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")

        self.detected_objects = None
        self.lock = threading.Lock()

        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects

        collision_points = np.array([], dtype=DTYPE)

        if len(msg.waypoints) == 0 or detected_objects is None:
            empty_msg = msgify(PointCloud2, collision_points, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            self.local_path_collision_pub.publish(empty_msg)
            return

        path_line = LineString([(p.position.x, p.position.y) for p in msg.waypoints])
        if path_line.is_empty:
            empty_msg = msgify(PointCloud2, collision_points, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            self.local_path_collision_pub.publish(empty_msg)
            return

        buffer_polygon = path_line.buffer(self.safety_box_width / 2.0, cap_style='flat')
        if buffer_polygon.is_empty:
            empty_msg = msgify(PointCloud2, collision_points, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            self.local_path_collision_pub.publish(empty_msg)
            return

        prepare(buffer_polygon)


        for obj in detected_objects:
            hull_floats = obj.convex_hull
            if len(hull_floats) % 3 != 0 or len(hull_floats) < 9:
                continue

            hull_points = [(hull_floats[i], hull_floats[i + 1]) for i in range(0, len(hull_floats), 3)]

            obj_polygon = Polygon(hull_points)
            if obj_polygon.is_empty or not obj_polygon.is_valid:
                continue

            if not buffer_polygon.intersects(obj_polygon):
                continue

            inter_geom = intersection(buffer_polygon, obj_polygon)
            intersection_points = get_coordinates(inter_geom)

            obj_speed = math.hypot(obj.velocity.x, obj.velocity.y)
            category = 3 if obj_speed < self.stopped_speed_limit else 4

            for x, y in intersection_points:
                collision_points = np.append(collision_points, np.array([
                    (x, y, obj.centroid.z,
                    obj.velocity.x, obj.velocity.y, obj.velocity.z,
                    self.braking_safety_distance_obstacle, np.inf, category)
                ], dtype=DTYPE))


        pc2_msg = msgify(PointCloud2, collision_points, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self.local_path_collision_pub.publish(pc2_msg)

        for pt in collision_points:
            print([
                f"x: {pt['x']}", f"y: {pt['y']}", f"z: {pt['z']}",
                f"vx: {pt['vx']}", f"vy: {pt['vy']}", f"vz: {pt['vz']}",
                f"distance_to_stop: {pt['distance_to_stop']}",
                f"deceleration_limit: {pt['deceleration_limit']}",
                f"category: {pt['category']}"
            ])

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()
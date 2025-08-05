#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header, Float32
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z', 'label']], dtype=np.float32)
        labels = data['label']

        if msg.header.frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
        
            tf_matrix = numpify(transform.transform).astype(np.float32)
            points = points.copy()
            points[:,3] = 1
            points = points.dot(tf_matrix.T)
        

        detected_msg = DetectedObjectArray()
        detected_msg.header.stamp = msg.header.stamp
        detected_msg.header.frame_id = self.output_frame

        unique_labels = np.unique(labels)

        valid_labels = []
        for label in unique_labels:
            if np.sum(labels == label) >= self.min_cluster_size:
                valid_labels.append(label)
        
        if len(valid_labels) == 0:
            rospy.loginfo("%s - No valid clusters found", rospy.get_name())
            self.objects_pub.publish(detected_msg)
            return
        
        for label in valid_labels:
            mask = (labels == label)
            cluster_points = points[mask, :3]

            obj = DetectedObject()
            obj.id = label
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            obj.position_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False

            centroid = np.mean(cluster_points, axis=0)
            obj.centroid.x = centroid[0]
            obj.centroid.y = centroid[1]
            obj.centroid.z = centroid[2]

            points_2d = MultiPoint(cluster_points[:, :2])
            hull = points_2d.convex_hull
            convex_hull_points = [a for hull in [[x, y, centroid[2]] for x, y in hull.exterior.coords] for a in hull]
            obj.convex_hull = convex_hull_points
                        
            detected_msg.objects.append(obj)

        self.objects_pub.publish(detected_msg)
        
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()
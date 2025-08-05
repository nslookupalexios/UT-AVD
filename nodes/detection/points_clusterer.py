#!/usr/bin/env python3

import rospy
import numpy as np
from sklearn.cluster import DBSCAN

from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify, msgify
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured


class PointsClusterer:
    def __init__(self):

        self.cluster_epsilon = rospy.get_param("~cluster_epsilon")
        self.cluster_min_size = rospy.get_param("~cluster_min_size")

        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)

        self.cluster_pub = rospy.Publisher(
            'points_clustered',
            PointCloud2,
            queue_size=1,
            tcp_nodelay=True
        )

        rospy.Subscriber(
            'points_filtered',
            PointCloud2,
            self.points_callback,
            queue_size=1,
            buff_size=2**24,
            tcp_nodelay=True
        )

    def points_callback(self, msg: PointCloud2):
        points_array = self.extract_xyz(msg)
        labels = self.clusterer.fit_predict(points_array)

        if points_array.shape[0] != labels.shape[0]:
            rospy.logwarn("The number of points and labels do not match!")
            return

        labeled_points = self.combine_points_and_labels(points_array, labels)
        structured_data = self.to_structured_array(labeled_points)

        cluster_msg = self.create_cluster_msg(structured_data, msg)
        self.cluster_pub.publish(cluster_msg)

    def extract_xyz(self, msg: PointCloud2) -> np.ndarray:
        data = numpify(msg)
        return structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)

    def combine_points_and_labels(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        labels = labels.reshape(-1, 1)
        points_with_labels = np.hstack((points, labels))
        return points_with_labels[labels[:, 0] != -1]

    def to_structured_array(self, points_labeled: np.ndarray) -> np.ndarray:
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
        ])
        return unstructured_to_structured(points_labeled, dtype=dtype)

    def create_cluster_msg(self, data: np.ndarray, original_msg: PointCloud2) -> PointCloud2:
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = original_msg.header.stamp
        cluster_msg.header.frame_id = original_msg.header.frame_id
        return cluster_msg

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    PointsClusterer().run()
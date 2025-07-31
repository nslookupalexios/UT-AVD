#!/usr/bin/env python3

import rospy
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest
from lanelet2 import traffic_rules, routing
from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import Waypoint


def load_lanelet2_map(map_path):
    """
    Load a Lanelet2 map using UTM projection parameters from the ROS parameter server.

    :param map_path: Path to the Lanelet2 map file
    :return: Loaded Lanelet2 map object
    """
    coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
    use_custom_origin = rospy.get_param("/localization/use_custom_origin")
    utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
    utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

    if coordinate_transformer != "utm":
        raise ValueError(f"Unsupported coordinate_transformer: {coordinate_transformer}. Only 'utm' is supported.")

    projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
    return load(map_path, projector)


class Lanelet2GlobalPlanner:
    def __init__(self):
        """
        Initialize the global planner: load map, create routing graph, and set up ROS publishers/subscribers.
        """
        # Load parameters
        map_path = rospy.get_param("~lanelet2_map_path")
        self.speed_limit = int(rospy.get_param("~speed_limit"))
        self.output_frame = rospy.get_param("lanelet2_global_planner/output_frame")
        self.distance_to_goal_limit = rospy.get_param("~distance_to_goal_limit", 1.0)


        # Internal state
        self.current_location = None
        self.goal_point = None

        # Load Lanelet2 map
        self.lanelet2_map = load_lanelet2_map(map_path)

        # Create traffic rules and routing graph
        self.traffic_rules = traffic_rules.create(
            traffic_rules.Locations.Germany,
            traffic_rules.Participants.VehicleTaxi
        )
        self.routing_graph = routing.RoutingGraph(self.lanelet2_map, self.traffic_rules)

        # ROS Publisher and Subscribers
        self.waypoints_pub = rospy.Publisher(
            'global_path', Path, queue_size=10, latch=True, tcp_nodelay=True
        )
        rospy.Subscriber(
            '/localization/current_pose', PoseStamped, self.current_pose_callback,
            queue_size=1, tcp_nodelay=True
        )
        rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self.goal_callback,
            queue_size=1, tcp_nodelay=True
        )

    def run(self):
        """Start the ROS node spin loop."""
        rospy.spin()

    def current_pose_callback(self, msg):
        """
        Callback to handle updates to the current vehicle pose.

        :param msg: ROS PoseStamped containing current pose
        """
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        if self.goal_point is not None:
            dist = ((self.current_location.x - self.goal_point.x)**2 + (self.current_location.y - self.goal_point.y)**2)**0.5
            if dist <= self.distance_to_goal_limit:
                rospy.loginfo("[%s] Goal reached! Distance = %.2f m. Clearing path.", rospy.get_name(), dist)
                self._clear_path()
                self.goal_point = None

    def goal_callback(self, msg):
        """
        Callback to handle reception of a new goal pose. Computes and publishes a path using Lanelet2 routing.

        :param msg: ROS PoseStamped containing goal pose
        """
        rospy.loginfo(
            "[%s] Received goal position: (%.3f, %.3f, %.3f), orientation: (%.3f, %.3f, %.3f, %.3f) in frame %s",
            rospy.get_name(),
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
            msg.header.frame_id
        )

        if self.current_location is None:
            rospy.logwarn("[%s] Current pose is not yet available.", rospy.get_name())
            return

        # Get nearest lanelets for start and goal points
        start_lanelet = self._get_nearest_lanelet(self.current_location, "start")
        if not start_lanelet:
            return

        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        goal_lanelet = self._get_nearest_lanelet(self.goal_point, "goal")
        if not goal_lanelet:
            return

        # Compute route using the routing graph
        route = self.routing_graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        if route is None:
            rospy.logwarn("[%s] No valid route found from start to goal lanelet.", rospy.get_name())
            return

        # Get the shortest path and extract the portion before a lane change is required
        shortest_path = route.shortestPath()
        continuous_path = shortest_path.getRemainingLane(start_lanelet)

        rospy.loginfo("[%s] Route successfully calculated with %d lanelets.", rospy.get_name(), len(continuous_path))

        self._publish_path(continuous_path)

    
    def _lanelet_seq_to_waypoints(self, lanelet_sequence):
        """
        Convert a LaneletSequence into a list of Waypoints.

        :param lanelet_sequence: Lanelet2 LaneletSequence
        :return: List of Waypoint messages
        """
        waypoints = []
        seen = set()

        for lanelet in lanelet_sequence:
            if 'speed_ref' in lanelet.attributes:
                speed_kmh = float(lanelet.attributes['speed_ref'])
            else:
                speed_kmh = self.speed_limit

            speed_mps = min(speed_kmh, self.speed_limit) * (1000.0 / 3600.0)

            for point in lanelet.centerline:
                point_id = (round(point.x, 3), round(point.y, 3), round(point.z, 3))
                if point_id in seen:
                    continue
                seen.add(point_id)

                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed_mps

                waypoints.append(waypoint)

        return waypoints

    def _publish_path(self, lanelet_sequence):
        """
        Convert a LaneletSequence to a Path message and publish it.

        :param lanelet_sequence: Lanelet2 LaneletSequence
        """
        waypoints = self._lanelet_seq_to_waypoints(lanelet_sequence)

        path = Path()
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints

        self.waypoints_pub.publish(path)

    
    def _get_nearest_lanelet(self, point, label):
        """
        Utility method to find the nearest lanelet to a given 2D point.

        :param point: BasicPoint2d instance
        :param label: String label used for logging ('start' or 'goal')
        :return: Closest lanelet object or None
        """
        try:
            nearest = findNearest(self.lanelet2_map.laneletLayer, point, 1)
            return nearest[0][1] if nearest else None
        except Exception as e:
            rospy.logerr("[%s] Error finding nearest lanelet to %s point: %s", rospy.get_name(), label, str(e))
            return None
    
    def _clear_path(self):
        """
        Publishes an empty path to clear the current route.
        """
        path = Path()
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = []

        self.waypoints_pub.publish(path)



if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    planner = Lanelet2GlobalPlanner()
    planner.run()

#!/usr/bin/env python3
import rospy
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest
from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped



def load_lanelet2_map(lanelet2_map_path):
    """
    Load a lanelet2 map from a file and return it
    :param lanelet2_map_path: name of the lanelet2 map file
    :param coordinate_transformer: coordinate transformer
    :param use_custom_origin: use custom origin
    :param utm_origin_lat: utm origin latitude
    :param utm_origin_lon: utm origin longitude
    :return: lanelet2 map
    """

    # get parameters
    coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
    use_custom_origin = rospy.get_param("/localization/use_custom_origin")
    utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
    utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

    # Load the map using Lanelet2
    if coordinate_transformer == "utm":
        projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
    else:
        raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

    lanelet2_map = load(lanelet2_map_path, projector)

    return lanelet2_map

class Lanelet2GlobalPlanner:
    def __init__(self):
        # Parameters
        #self.output_frame = rospy.get_param("output_frame")
        #self.distance_to_goal_limit = rospy.get_param("distance_to_goal_limit")
        #self.distance_to_centerline_limit = rospy.get_param("~distance_to_centerline_limit")
        #self.speed_limit = rospy.get_param("speed_limit")
        #self.ego_vehicle_stopped_speed_limit = rospy.get_param("ego_vehicle_stopped_speed_limit")
        #self.lane_change = rospy.get_param("~lane_change")
        #self.lanelet_search_radius = rospy.get_param("~lanelet_search_radius")
        #self.lane_change_base_length = rospy.get_param("lane_change_base_length")
        #self.lane_change_perlane_length = rospy.get_param("lane_change_perlane_length")
        #self.waypoint_interval = rospy.get_param("waypoint_interval")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")
        #self.routing_cost = rospy.get_param("~routing_cost")

        # Internal variables
        self.lanelet_candidates = []
        self.current_location = None
        self.current_speed = None
        self.goal_point = None

        self.lanelet2_map = load_lanelet2_map(lanelet2_map_path)


        # routing graph
        #self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules, routing_costs)

        # Publishers
        #self.waypoints_pub = rospy.Publisher('lanelet2_global_path', Path, queue_size=10, latch=True, tcp_nodelay=True)
        #self.target_lane_pub = rospy.Publisher('target_lane_markers', MarkerArray, queue_size=10, latch=True, tcp_nodelay=True)

        # Subscribers
        #rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        #rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=None, tcp_nodelay=True)

    def run(self):
        rospy.spin()
    
    def goal_callback(self, msg):
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        return































        if self.current_location is None:
            # TODO handle if current_pose gets lost at later stage - see current_pose_callback
            rospy.logwarn("%s - current_pose not available", rospy.get_name())
            return

        # Using current pose as start point
        start_point = shapely.Point(self.current_location.x, self.current_location.y)
        # Get nearest lanelets to start point
        start_lanelet_candidates = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(start_point.x, start_point.y), self.lanelet_search_radius)
        # If no lanelet found near start point, return
        if len(start_lanelet_candidates) == 0:
            rospy.logerr("%s - no lanelet found near start point", rospy.get_name())
            return
        # Extract lanelet objects from candidates
        start_lanelet_candidates = [start_lanelet[1] for start_lanelet in start_lanelet_candidates]
        lanelet_candidates = [start_lanelet_candidates] + self.lanelet_candidates[1:]

        new_goal = shapely.Point(msg.pose.position.x, msg.pose.position.y)
        # Get nearest lanelets to goal point
        goal_lanelet_candidates = findWithin2d(self.lanelet2_map.laneletLayer, BasicPoint2d(new_goal.x, new_goal.y), self.lanelet_search_radius)
        # If no lanelet found near goal point, return
        if len(goal_lanelet_candidates) == 0:
            rospy.logerr("%s - no lanelet found near goal point", rospy.get_name())
            return
        # Extract lanelet objects from candidates
        goal_lanelet_candidates = [goal_lanelet[1] for goal_lanelet in goal_lanelet_candidates]
        # Add current goal candidates to lanelet candidates list
        lanelet_candidates.append(goal_lanelet_candidates)

        # Find shortest path and shortest route
        route, stops = self.get_shortest_route(lanelet_candidates)
        if route is None:
            rospy.logerr("%s - no route found, try new goal!", rospy.get_name())
            return
        path = route.shortestPath()
        if path is None:
            rospy.logerr("%s - no path found, try new goal!", rospy.get_name())
            return

        # Publish target lanelets for visualization
        start_lanelet = path[0]
        goal_lanelet = path[-1]
        self.publish_target_lanelets(start_lanelet, goal_lanelet)

        # Convert lanelet path to waypoints
        waypoints = self.convert_to_waypoints(path, route)
        if waypoints is None:
            rospy.logerr("%s - route contained an impossible lane change!", rospy.get_name())
            return

        global_path = PathWrapper(waypoints, velocities=True, blinkers=True)

        # Find distance to start and goal waypoints
        start_point_distance = global_path.linestring.project(start_point)
        new_goal_point_distance = global_path.linestring.project(new_goal)

        # Interpolate point coordinates
        start_on_path = global_path.linestring.interpolate(start_point_distance)
        new_goal_on_path = global_path.linestring.interpolate(new_goal_point_distance)

        if shapely.distance(start_on_path, start_point) > self.distance_to_centerline_limit:
            rospy.logerr("%s - start point too far from centerline", rospy.get_name())
            return

        if shapely.distance(new_goal_on_path, new_goal) > self.distance_to_centerline_limit:
            rospy.logerr("%s - goal point too far from centerline", rospy.get_name())
            return

        if start_lanelet.id == goal_lanelet.id and start_point_distance > new_goal_point_distance:
            rospy.logerr("%s - goal point can't be on the same lanelet before start point", rospy.get_name())
            return

        # If there is only one goal candidate, we can fix the preceding lanelets to be the stops on the best found route
        if len(goal_lanelet_candidates) == 1:
            lanelet_candidates = [[lanelet] for lanelet in stops]

        # Update member variables
        self.goal_point = new_goal_on_path
        self.lanelet_candidates = lanelet_candidates
        rospy.logdebug("Lanelet candidates: " + str(list(map(len, lanelet_candidates))))

        # Trim the global path 
        trimmed_waypoints = global_path.extract_waypoints(start_point_distance, new_goal_point_distance, trim=True, copy=True)

        # Publish the global path
        self.publish_waypoints(trimmed_waypoints)
        rospy.loginfo("%s - global path published", rospy.get_name())


if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()
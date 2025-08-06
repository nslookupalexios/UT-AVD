#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import threading
import tf2_ros
import onnxruntime
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from image_geometry import PinholeCameraModel
from shapely.geometry import LineString

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from autoware_mini.msg import TrafficLightResult, TrafficLightResultArray
from autoware_mini.msg import Path
from tf2_geometry_msgs import do_transform_point

from cv_bridge import CvBridge, CvBridgeError

# Classifier outputs 4 classes (LightState)
CLASSIFIER_RESULT_TO_STRING = {
    0: "green",
    1: "yellow",
    2: "red",
    3: "unknown"
}

CLASSIFIER_RESULT_TO_COLOR = {
    0: (0, 255, 0),
    1: (255, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 0)
}

CLASSIFIER_RESULT_TO_TLRESULT = {
    0: 1,  # GREEN
    1: 0,  # YELLOW
    2: 0,  # RED
    3: 2  # UNKNOWN
}


class CameraTrafficLightDetector:
    def __init__(self):

        # Node parameters
        onnx_path = rospy.get_param("~onnx_path")
        self.rectify_image = rospy.get_param('~rectify_image')
        self.roi_width_extent = rospy.get_param("~roi_width_extent")
        self.roi_height_extent = rospy.get_param("~roi_height_extent")
        self.min_roi_width = rospy.get_param("~min_roi_width")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        # Parameters related to lanelet2 map loading
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        # global variables
        self.trafficlights = None
        self.tfl_stoplines = None
        self.camera_model = None
        self.stoplines_on_path = None

        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        # Extract all stop lines and signals from the lanelet2 map
        all_stoplines = get_stoplines(lanelet2_map)
        self.trafficlights = get_stoplines_trafficlights(lanelet2_map)
        # If stopline_id is not in self.signals then it has no signals (traffic lights)
        self.tfl_stoplines = {k: v for k, v in all_stoplines.items() if k in self.trafficlights}

        # Publishers
        self.tfl_status_pub = rospy.Publisher('traffic_light_status', TrafficLightResultArray, queue_size=1,
                                              tcp_nodelay=True)
        self.tfl_roi_pub = rospy.Publisher('traffic_light_roi', Image, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('camera_info', CameraInfo, self.camera_info_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/planning/local_path', Path, self.local_path_callback, queue_size=1, buff_size=2 ** 20,
                         tcp_nodelay=True)
        rospy.Subscriber('image_raw', Image, self.camera_image_callback, queue_size=1, buff_size=2 ** 26,
                         tcp_nodelay=True)

    def camera_info_callback(self, camera_info_msg):
        if self.camera_model is None:
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(camera_info_msg)
            self.camera_model = camera_model

    def local_path_callback(self, local_path_msg):
        stoplines_on_path = []

        if not local_path_msg.waypoints:
            rospy.logwarn_throttle(10, "%s - Received empty local path", rospy.get_name())
            return

        path_points = [(pose.position.x, pose.position.y) for pose in local_path_msg.waypoints]
        path_line = LineString(path_points)

        for stopline_id, stopline in self.tfl_stoplines.items():
            if path_line.intersects(stopline):
                stoplines_on_path.append(stopline_id)

        with self.lock:
            self.stoplines_on_path = stoplines_on_path
            self.transform_from_frame = local_path_msg.header.frame_id


    def camera_image_callback(self, camera_image_msg):
        if self.camera_model is None:
            rospy.logwarn_throttle(10, "%s - No camera model received, skipping image", rospy.get_name())
            return

        if self.stoplines_on_path is None:
            rospy.logwarn_throttle(10, "%s - No path received, skipping image", rospy.get_name())
            return
        
        if self.transform_from_frame is None:
            rospy.logwarn_throttle(10, "%s - No valid transform from this frame, skipping image", rospy.get_name())
            return
        
        rospy.loginfo("stoplines_on_path: %s", str(self.stoplines_on_path))
        
        try:
            transform = self.tf_buffer.lookup_transform(
                camera_image_msg.header.frame_id,
                self.transform_from_frame,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as ex:
            rospy.logwarn_throttle(10, "[%s] TF lookup failed: %s", rospy.get_name(), str(ex))
            return

        
        rois = self.calculate_roi_coordinates(self.stoplines_on_path, transform)
        rospy.loginfo("rois: %s", str(rois))

        try:
            # Convert the ROS image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(camera_image_msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: %s", str(e))
            return

        # Rectify image if requested
        if self.rectify_image:
            self.camera_model.rectifyImage(image, image)

        
        if rois:
            roi_images = self.create_roi_images(image, rois)

            if roi_images is None:
                rospy.logwarn_throttle(10, "[%s] No valid ROI images created", rospy.get_name())
                return

            # Inference
            predictions = self.model.run(None, {'conv2d_1_input': roi_images})[0]
            rospy.loginfo("predictions: %s", str(predictions))


        # Publish the image as-is with empty bounding boxes
        empty_rois = []
        empty_classes = []
        empty_scores = []
        self.publish_roi_images(image, empty_rois, empty_classes, empty_scores, camera_image_msg.header.stamp)

        # Publish an empty TrafficLightResultArray
        tfl_status = TrafficLightResultArray()
        tfl_status.header.stamp = camera_image_msg.header.stamp
        self.tfl_status_pub.publish(tfl_status)


    def calculate_roi_coordinates(self, stoplines_on_path, transform):
        rois = []

        for linkId in stoplines_on_path:
            for plId, traffic_lights in self.trafficlights[linkId].items():
                us = []
                vs = []

                for x, y, z in traffic_lights.values():
                    point_map = Point(float(x), float(y), float(z))

                    try:
                        point_stamped = PointStamped()
                        point_stamped.header.stamp = rospy.Time(0)
                        point_stamped.header.frame_id = transform.header.frame_id
                        point_stamped.point = point_map

                        point_camera = do_transform_point(point_stamped, transform).point

                        # Coordinate pixel da punto 3D
                        u, v = self.camera_model.project3dToPixel(
                            (point_camera.x, point_camera.y, point_camera.z))

                        # Ignora se fuori dall'immagine
                        if u < 0 or u >= self.camera_model.width or v < 0 or v >= self.camera_model.height:
                            continue

                        # Estensione in pixel (da metri)
                        extent_x_px = self.camera_model.fx() * self.roi_width_extent / point_camera.z
                        extent_y_px = self.camera_model.fy() * self.roi_height_extent / point_camera.z

                        us.extend([u + extent_x_px, u - extent_x_px])
                        vs.extend([v + extent_y_px, v - extent_y_px])

                    except Exception as e:
                        rospy.logwarn_throttle(10, "[%s] ROI projection failed: %s", rospy.get_name(), str(e))
                        continue

                # Se meno di 4 vertici validi (quindi meno di 8 coordinate), ignora
                if len(us) < 8:
                    continue

                # Clipping e rounding
                us = np.clip(np.round(np.array(us)), 0, self.camera_model.width - 1)
                vs = np.clip(np.round(np.array(vs)), 0, self.camera_model.height - 1)

                min_u = int(np.min(us))
                max_u = int(np.max(us))
                min_v = int(np.min(vs))
                max_v = int(np.max(vs))

                # Scarta bounding box troppo piccole
                if max_u - min_u < self.min_roi_width:
                    continue

                # Aggiungi ROI
                rois.append([int(linkId), plId, min_u, max_u, min_v, max_v])

        return rois


    def create_roi_images(self, image, rois):
        roi_images = []

        for _, _, min_u, max_u, min_v, max_v in rois:
            try:
                # Ritaglio ROI dall'immagine
                roi = image[min_v:max_v, min_u:max_u, :]

                # Ridimensionamento a 128x128
                roi_resized = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)

                # Conversione a float32 e normalizzazione
                roi_images.append(roi_resized.astype(np.float32))

            except Exception as e:
                rospy.logwarn_throttle(10, "[%s] ROI creation failed: %s", rospy.get_name(), str(e))
                continue

        # Stacka tutte le immagini lungo l'asse 0 e normalizza [0,1]
        return np.stack(roi_images, axis=0) / 255.0 if roi_images else None


    def publish_roi_images(self, image, rois, classes, scores, image_time_stamp):

        # add rois to image
        if len(rois) > 0:
            for cl, score, (_, _, min_u, max_u, min_v, max_v) in zip(classes, scores, rois):
                text_string = "%s %.2f" % (CLASSIFIER_RESULT_TO_STRING[cl], score)
                text_width, text_height = cv2.getTextSize(text_string, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                text_orig_u = int(min_u + (max_u - min_u) / 2 - text_width / 2)
                text_orig_v = max_v + text_height + 3

                start_point = (min_u, min_v)
                end_point = (max_u, max_v)
                cv2.rectangle(image, start_point, end_point, color=CLASSIFIER_RESULT_TO_COLOR[cl], thickness=3)
                cv2.putText(image,
                            text_string,
                            org=(text_orig_u, text_orig_v),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5,
                            color=CLASSIFIER_RESULT_TO_COLOR[cl],
                            thickness=2)

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')

        img_msg.header.stamp = image_time_stamp
        self.tfl_roi_pub.publish(img_msg)

    def run(self):
        rospy.spin()


def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 map
    :return: {stop_line_id: stopline, ...}
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stoline to dictionary and convert it to shapely LineString
                stoplines[line.id] = LineString([(p.x, p.y) for p in line])

    return stoplines


def get_stoplines_trafficlights(lanelet2_map):
    """
    Iterate over all regulatory_elements with subtype traffic light and extract the stoplines and sinals.
    Organize the data into dictionary indexed by stopline id that contains a traffic_light id and the four coners of the traffic light.
    :param lanelet2_map: lanelet2 map
    :return: {stopline_id: {traffic_light_id: {'top_left': [x, y, z], 'top_right': [...], 'bottom_left': [...], 'bottom_right': [...]}, ...}, ...}
    """

    signals = {}

    for reg_el in lanelet2_map.regulatoryElementLayer:
        if reg_el.attributes["subtype"] == "traffic_light":
            # ref_line is the stop line and there is only 1 stopline per traffic light reg_el
            linkId = reg_el.parameters["ref_line"][0].id

            for tfl in reg_el.parameters["refers"]:
                tfl_height = float(tfl.attributes["height"])
                # plId represents the traffic light (pole), one stop line can be associated with multiple traffic lights
                plId = tfl.id

                traffic_light_data = {'top_left': [tfl[0].x, tfl[0].y, tfl[0].z + tfl_height],
                                      'top_right': [tfl[1].x, tfl[1].y, tfl[1].z + tfl_height],
                                      'bottom_left': [tfl[0].x, tfl[0].y, tfl[0].z],
                                      'bottom_right': [tfl[1].x, tfl[1].y, tfl[1].z]}

                # signals is a dictionary indexed by stopline id and contains dictionary of traffic lights indexed by pole id
                # which in turn contains a dictionary of traffic light corners
                signals.setdefault(linkId, {}).setdefault(plId, traffic_light_data)

    return signals

if __name__ == '__main__':
    rospy.init_node('camera_traffic_light_detector', log_level=rospy.INFO)
    node = CameraTrafficLightDetector()
    node.run()
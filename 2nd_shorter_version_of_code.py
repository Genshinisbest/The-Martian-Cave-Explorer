# #!/usr/bin/env python3

# # Updated cave_explorer with corrected Planning 1 → 2 → 3 behaviour order.

# import math
# import random
# from enum import Enum
# import os
# import datetime
# import time
# import numpy as np

# try:
#     import torch
# except ImportError:
#     torch = None

# import cv2
# import rclpy
# from cv_bridge import CvBridge
# from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
# from nav2_msgs.action import NavigateToPose
# from nav_msgs.msg import OccupancyGrid
# from rclpy.action import ActionClient
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from tf2_ros import TransformException
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener
# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray

# def wrap_angle(angle):
#     while angle < 0.0:
#         angle += 2 * math.pi
#     while angle > 2 * math.pi:
#         angle -= 2 * math.pi
#     return angle

# def pose2d_to_pose(pose_2d):
#     pose = Pose()
#     pose.position.x = pose_2d.x
#     pose.position.y = pose_2d.y
#     pose.orientation.w = math.cos(pose_2d.theta / 2.0)
#     pose.orientation.z = math.sin(pose_2d.theta / 2.0)
#     return pose

# class PlannerType(Enum):
#     ERROR = 0
#     MOVE_FORWARDS = 1
#     RETURN_HOME = 2
#     GO_TO_FIRST_ARTIFACT = 3
#     RANDOM_WALK = 4
#     RANDOM_GOAL = 5
#     FRONTIER_EXPLORATION = 6
#     INSPECT_ARTIFACT = 7

# class CaveExplorer(Node):
#     def __init__(self):
#         super().__init__('cave_explorer_node')
#         self.xlim_ = [0.0, 0.0]
#         self.ylim_ = [0.0, 0.0]
#         self.artifact_found_ = False
#         self.save_interval = 3.0
#         self.last_saved_time = 0.0
#         self.image_counter = 0
#         self.custom_model = None
#         self.use_custom_model = False
#         try:
#             if torch is not None:
#                 model_path = '/home/student/ros_ws/CV_Model/yolov5/best.pt'
#                 repo_path = '/home/student/ros_ws/CV_Model/yolov5'
#                 if os.path.exists(model_path) and os.path.exists(repo_path):
#                     import sys
#                     cwd = os.getcwd()
#                     os.chdir(repo_path)
#                     sys.path.insert(0, repo_path)
#                     self.custom_model = torch.hub.load('.', 'custom', path='best.pt', source='local', trust_repo=True)
#                     self.use_custom_model = True
#                     os.chdir(cwd)
#                     self.get_logger().info('Loaded custom model via torch.hub')
#             else:
#                 self.get_logger().warn('Torch not available')
#         except Exception as e:
#             self.get_logger().error(f'Model load error: {e}')
#             self.use_custom_model = False

#         self.detected_artifacts_positions = []
#         self.artifact_clustering_threshold = 2.5
#         self.planner_type_ = PlannerType.ERROR
#         self.reached_first_artifact_ = False
#         self.returned_home_ = False
#         self.current_map_ = None
#         self.map_info_ = None
#         self.exploration_complete_ = False
#         self.artifacts_to_inspect_ = []
#         self.inspected_artifacts_ = []
#         self.currently_inspecting_ = False
#         self.inspect_timeout_s = 30.0
#         self.inspect_start_time = None
#         self.marker_artifacts_ = Marker()
#         self.marker_artifacts_.header.frame_id = 'map'
#         self.marker_artifacts_.ns = 'artifacts'
#         self.marker_artifacts_.id = 0
#         self.marker_artifacts_.type = Marker.SPHERE_LIST
#         self.marker_artifacts_.action = Marker.ADD
#         self.marker_artifacts_.scale.x = 1.5
#         self.marker_artifacts_.scale.y = 1.5
#         self.marker_artifacts_.scale.z = 1.5
#         self.marker_artifacts_.color.a = 1.0
#         self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)
#         self.artifact_locations_ = []
#         self.cv_bridge_ = CvBridge()
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
#         self.get_logger().warn('Waiting for navigate_to_pose action...')
#         self.nav2_action_client_.wait_for_server()
#         self.get_logger().warn('navigate_to_pose connected')
#         self.ready_for_next_goal_ = True
#         self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)
#         self.declare_parameter('initial_mode', 'random')
#         self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)
#         self.map_sub_ = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
#         self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
#         self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)
#         self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

#     def get_pose_2d(self):
#         try:
#             t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
#         except TransformException:
#             return None
#         pose = Pose2D()
#         pose.x = t.transform.translation.x
#         pose.y = t.transform.translation.y
#         qw = t.transform.rotation.w
#         qz = t.transform.rotation.z
#         pose.theta = wrap_angle(2. * math.acos(qw) if qz >= 0 else -2. * math.acos(qw))
#         return pose

#     def map_callback(self, map_msg: OccupancyGrid):
#         map_origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
#         map_resolution = map_msg.info.resolution
#         map_height = map_msg.info.height
#         map_width = map_msg.info.width
#         self.xlim_ = [map_origin[0], map_origin[0] + map_width * map_resolution]
#         self.ylim_ = [map_origin[1], map_origin[1] + map_height * map_resolution]
#         self.current_map_ = np.array(map_msg.data).reshape((map_height, map_width))
#         self.map_info_ = map_msg.info

#     def image_callback(self, image_msg):
#         image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
#         self.save_images_for_dataset(image)
#         detections = self.detect_artifacts_advanced(image)
#         artifact_positions = []
#         for det in detections:
#             x, y, w, h, conf, atype = det
#             if conf >= 0.35:
#                 cx = x + w // 2
#                 cy = y + h // 2
#                 pos3 = self.get_artifact_3d_position(cx, cy, h)
#                 if pos3:
#                     artifact_positions.append((pos3, atype, conf))
#         self.artifact_found_ = len(artifact_positions) > 0
#         for pos3, atype, conf in artifact_positions:
#             self.add_detected_artifact(pos3, atype, conf)
#         for det in detections:
#             x, y, w, h, conf, atype = det
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, f"{atype}:{conf:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#         msg = self.cv_bridge_.cv2_to_imgmsg(image, encoding='rgb8')
#         self.image_detections_pub_.publish(msg)

#     def save_images_for_dataset(self, image):
#         now = time.time()
#         if now - self.last_saved_time >= self.save_interval:
#             save_dir = '/home/student/ros_ws/dataset_images'
#             os.makedirs(save_dir, exist_ok=True)
#             fname = f"img_{self.image_counter:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#             path = os.path.join(save_dir, fname)
#             cv2.imwrite(path, image)
#             self.image_counter += 1
#             self.last_saved_time = now

#     def detect_artifacts_advanced(self, image):
#         detections = []
#         if self.use_custom_model and self.custom_model:
#             results = self.custom_model(image)
#             model_dets = results.xyxy[0].cpu().numpy()
#             labels = ['green_crystal', 'green_alien', 'stop_sign', 'mushrooms', 'formation', 'white_sphere']
#             for d in model_dets:
#                 x1, y1, x2, y2, conf, cls = d
#                 if conf >= 0.1:
#                     cid = int(cls)
#                     if cid < len(labels):
#                         detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1), float(conf), labels[cid]))
#         return detections

#     def get_artifact_3d_position(self, pixel_x, pixel_y, box_height):
#         robot_pose = self.get_pose_2d()
#         if not robot_pose:
#             return None
#         image_width = 640
#         fov_deg = 70.0
#         offset_x = pixel_x - (image_width / 2)
#         direction = robot_pose.theta + (offset_x / image_width) * math.radians(fov_deg)
#         est_dist = max(0.8, min(12.0, 220.0 / float(box_height))) if box_height > 0 else 4.0
#         ax = robot_pose.x + est_dist * math.cos(direction)
#         ay = robot_pose.y + est_dist * math.sin(direction)
#         return (ax, ay, 0.5)

#     def add_detected_artifact(self, pos3, atype, conf):
#         if pos3 is None:
#             return
#         p = Point(); p.x, p.y, p.z = pos3
#         already = any(math.hypot(p.x - q.x, p.y - q.y) < 1.0 for q in self.artifacts_to_inspect_ + self.inspected_artifacts_)
#         if not already:
#             self.artifacts_to_inspect_.append(p)
#             self.get_logger().info(f'Queued artifact {atype} at ({p.x:.2f},{p.y:.2f})')
#         self.artifact_locations_.append(p)
#         self.publish_artifact_markers()

#     def publish_artifact_markers(self):
#         marray = MarkerArray()
#         m = self.marker_artifacts_
#         m.points = self.artifact_locations_
#         marray.markers.append(m)
#         self.marker_pub_.publish(marray)

#     def planner_go_to_pose2d(self, pose2d):
#         goal = NavigateToPose.Goal()
#         goal.pose.header.stamp = self.get_clock().now().to_msg()
#         goal.pose.header.frame_id = 'map'
#         goal.pose.pose = pose2d_to_pose(pose2d)
#         self.goal_pose_vis_.publish(goal.pose)
#         fb = self.feedback_callback if self.get_parameter('print_feedback').value else None
#         self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
#         self.send_goal_future_ = self.nav2_action_client_.send_goal_async(goal, feedback_callback=fb)
#         self.send_goal_future_.add_done_callback(self.goal_response_callback)

#     def goal_response_callback(self, future):
#         gh = future.result()
#         if not gh.accepted:
#             self.ready_for_next_goal_ = True
#             return
#         self.get_result_future_ = gh.get_result_async()
#         self.get_result_future_.add_done_callback(self.goal_reached_callback)

#     def feedback_callback(self, fb_msg):
#         fb = fb_msg.feedback
#         self.get_logger().info(f'{fb.distance_remaining:.2f} m remaining')

#     def goal_reached_callback(self, future):
#         self.ready_for_next_goal_ = True

#     def planner_random_goal(self):
#         random_goals = [[15.2, 2.2],[30.7, 2.2],[43.0, 11.3],[10.0,10.0]]
#         gx, gy = random.choice(random_goals)
#         pose = Pose2D(x=gx, y=gy, theta=random.uniform(0, 2*math.pi))
#         self.planner_go_to_pose2d(pose)

#     def planner_inspect_artifact(self):
#         if not self.artifacts_to_inspect_:
#             self.ready_for_next_goal_ = True
#             return
#         target = self.artifacts_to_inspect_[0]
#         robot_pose = self.get_pose_2d()
#         if not robot_pose:
#             self.ready_for_next_goal_ = True
#             return
#         dx = target.x - robot_pose.x
#         dy = target.y - robot_pose.y
#         angle = math.atan2(dy, dx)
#         ax = target.x - 1.0 * math.cos(angle)
#         ay = target.y - 1.0 * math.sin(angle)
#         self.currently_inspecting_ = True
#         self.inspect_start_time = time.time()
#         self.planner_go_to_pose2d(Pose2D(x=ax, y=ay, theta=angle))

#     def main_loop(self):
#         if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
#             self.get_logger().warn('Waiting for transform...')
#             return
#         if not self.ready_for_next_goal_:
#             if self.currently_inspecting_ and self.inspect_start_time and time.time() - self.inspect_start_time > self.inspect_timeout_s:
#                 self.currently_inspecting_ = False
#                 if self.artifacts_to_inspect_:
#                     self.artifacts_to_inspect_.pop(0)
#                     self.ready_for_next_goal_ = True
#             return
#         if self.currently_inspecting_ and self.ready_for_next_goal_:
#             if self.artifacts_to_inspect_:
#                 v = self.artifacts_to_inspect_.pop(0)
#                 self.inspected_artifacts_.append(v)
#             self.currently_inspecting_ = False

#         # --- Corrected behaviour order ---
#         initial_mode = self.get_parameter('initial_mode').value
#         if self.artifacts_to_inspect_:
#             self.planner_type_ = PlannerType.INSPECT_ARTIFACT
#         elif not self.exploration_complete_ and initial_mode == 'random':
#             self.planner_type_ = PlannerType.RANDOM_GOAL
#         else:
#             self.planner_type_ = PlannerType.FRONTIER_EXPLORATION

#         self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
#         if self.planner_type_ == PlannerType.INSPECT_ARTIFACT:
#             self.planner_inspect_artifact()
#         elif self.planner_type_ == PlannerType.RANDOM_GOAL:
#             self.planner_random_goal()
#         else:
#             self.get_logger().warn('No valid planner chosen')

# def main(args=None):
#     rclpy.init(args=args)
#     node = CaveExplorer()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

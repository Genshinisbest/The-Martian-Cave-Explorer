# #!/usr/bin/env python3
# import rospy
# import roslib
# import math
# import cv2 # OpenCV2
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from nav_msgs.srv import GetMap
# from nav_msgs.msg import OccupancyGrid
# import tf
# from std_srvs.srv import Empty
# from geometry_msgs.msg import Twist
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from geometry_msgs.msg import Pose2D
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from sensor_msgs.msg import Image, LaserScan # For Planning
# from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
# import actionlib
# import random
# import copy
# from threading import Lock
# from enum import Enum
# #Perception
# import os
# import datetime
# import time
# import torch #For running the CV model!
# #Advanced 1
# from sensor_msgs.msg import PointCloud2, PointField
# from sensor_msgs import point_cloud2
# from visualization_msgs.msg import Marker
# from sensor_msgs.msg import CameraInfo


# def wrap_angle(angle):
#     # Function to wrap an angle between 0 and 2*Pi
#     while angle < 0.0:
#         angle = angle + 2 * math.pi

#     while angle > 2 * math.pi:
#         angle = angle - 2 * math.pi

#     return angle

# def pose2d_to_pose(pose_2d):
#     pose = Pose()

#     pose.position.x = pose_2d.x
#     pose.position.y = pose_2d.y

#     pose.orientation.w = math.cos(pose_2d.theta / 2.0)
#     pose.orientation.z = math.sin(pose_2d.theta / 2.0)

#     return pose

# class ExplorationState(Enum):
#     EXPLORING = 1
#     INSPECTING = 2
#     IDLE = 3

# class PlannerType(Enum):
#     ERROR = 0
#     MOVE_FORWARDS = 1
#     RETURN_HOME = 2
#     GO_TO_FIRST_ARTIFACT = 3
#     RANDOM_WALK = 4
#     RANDOM_GOAL = 5
#     # Add more!
#     FRONTIER_EXPLORATION = 6
#     CLOSE_RANGE_INSPECTION = 7

# class CaveExplorer:
#     def __init__(self):

#         # Variables/Flags for perception
#         self.localised_ = False
#         self.artifact_found_ = False

#         # Variables/Flags for planning
#         self.planner_type_ = PlannerType.ERROR
#         self.reached_first_artifact_ = False
#         self.reached_frontier_goal_ = False
#         self.returned_home_ = False
#         self.goal_counter_ = 0 # gives each goal sent to move_base a unique ID
#         self.frontiers = []

#         # Image saving variables
#         self.last_saved_time = 0
#         self.save_interval = 1      # Second

#         # Camera intrinsic variables
#         self.fx = None
#         self.fy = None
#         self.cx = None
#         self.cy = None
#         self.current_pointcloud = []  

#         # Initialise CvBridge
#         self.cv_bridge_ = CvBridge()

#         # Wait for the transform to become available
#         rospy.loginfo("Waiting for transform from map to base_link")
#         self.tf_listener_ = tf.TransformListener()

#         while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
#             rospy.sleep(0.1)
#             print("Waiting for transform... Have you launched a SLAM node?")        

#         # Advertise "cmd_vel" publisher to control the robot manually -- though usually we will be controller via the following action client
#         self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

#         # Action client for move_base
#         self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
#         rospy.loginfo("Waiting for move_base action...")
#         self.move_base_action_client_.wait_for_server()
#         rospy.loginfo("move_base connected")

#         # Publisher for the camera detections
#         self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

#         # Read in computer vision model YOLOv5
#         self.computer_vision_model_ = torch.hub.load('/home/student/catkin_ws/CV_Model/yolov5', 'custom', path='/home/student/catkin_ws/CV_Model/yolov5/best.pt', source='local')

#         # Publish artifact detections
#         self.artifact_marker_pub_ = rospy.Publisher('marker', Marker, queue_size=20)

#         # Subscribe to the camera topic
#         self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

#         # Subscribe to the laser topic
#         self.laser_sub_ = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)

#         # Subscribe to the map for occupancy grid
#         self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
#         self.map_ = None
#         self.map_lock_ = Lock()

#         # Subscribe to the point cloud
#         self.pointcloud_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pointcloud_callback)

#         # Subscribe to camera info topic
#         self.caminfo_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)

#         # Plannin 3 - Behaviour Switching
#         self.inspected_artifacts = []
#         self.inspection_threshold = 2.0  # Distance threshold to consider an artifact as already inspected   
#         self.artifact_id_counter = 0 
#         self.artifact_position_ = None
#         self.detected_artifacts = []
#         self.state_ = ExplorationState.EXPLORING

#     def get_pose_2d(self):

#         # Lookup the latest transform
#         (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

#         # Return a Pose2D message
#         pose = Pose2D()
#         pose.x = trans[0]
#         pose.y = trans[1]

#         qw = rot[3];
#         qz = rot[2];

#         if qz >= 0.:
#             pose.theta = wrap_angle(2. * math.acos(qw))
#         else: 
#             pose.theta = wrap_angle(-2. * math.acos(qw));

#         print("pose: ", pose)

#         return pose

#     # Get point cloud data for Advanced 1
#     def pointcloud_callback(self, msg):
#         points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
#         self.current_pointcloud = msg 
#         #rospy.loginfo(f'Received {len(points)} points from point cloud.')

    
#     def camera_info_callback(self, msg):
#         #rospy.loginfo('Camera info callback')
#         self.fx = msg.K[0]
#         self.fy = msg.K[4]
#         self.cx = msg.K[2]
#         self.cy = msg.K[5]
#         #print(f"Focal length (fx, fy): ({self.fx}, {self.fy})")
#         #print(f"Principal point (cx, cy): ({self.cx}, {self.cy})")
         
#     # Get map data for Occupancy Grid
#     def map_callback(self, map_msg):
#         try:
#             with self.map_lock_:
#                 self.map_ = map_msg
#                 self.map_received = True
#                 # Debugging
#                 #unexplored_count = sum(1 for value in self.map_.data if value == -1)
#                 #rospy.loginfo(f'Number of unexplored cells: {unexplored_count}')
#         except Exception as e:
#             rospy.logerr(f'Error processing map data: {str(e)}')

#     # Get laserScan data 
#     def laser_callback(self, scan_msg):
#         # Get laser data
#         ranges = np.array(scan_msg.ranges)
#         angle_min = scan_msg.angle_min
#         angle_increment = scan_msg.angle_increment

#     def image_callback(self, image_msg):  
#         if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
#             rospy.loginfo('Camera intrinsics not available yet.')
#             return None
        
#         # Copy the image message to a cv image
#         image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

#         # Make sure writable
#         image.setflags(write=1)

#         # Saving image for dataset
#         current_time = time.time()
#         if current_time - self.last_saved_time >= self.save_interval:
#             save_directory = "/home/student/catkin_ws/robot_images"
#             if not os.path.exists(save_directory):
#                 os.makedirs(save_directory)

#             # Name file
#             filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
#             filepath = os.path.join(save_directory, filename)
#             cv2.imwrite(filepath, image)
#             self.last_saved_time = current_time

#         # Run YOLOv5 model on the image
#         results = self.computer_vision_model_(image)
#         detections = results.xyxy[0].cpu().numpy()

#         artifact_labels = ['green crystal', 'green alien', 'stop sign', 'mushrooms', 'formation', 'ice wall', 'white sphere']
        
#         bbox_centers = []
#         self.artifact_found_ = False  # Reset for this callback

#         # Detect artifacts and draw bounding boxes
#         for detection in detections:
#             x1, y1, x2, y2, conf, cls = detection
#             if conf >= 0.75:  # Only draw if confidence is greater than 75%
#                 class_id = int(cls)
#                 cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
#                 # Add label for class and confidence
#                 label = f"{artifact_labels[class_id]}: {conf:.2f}"
#                 cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # Center of boundary box
#                 bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
#                 bbox_centers.append((bbox_center, class_id))

#         # Handle averaged positions
#         averaged_positions = self.process_detections(bbox_centers)

#         # Handle averaged positions and find 3D artifacts
#         for class_id, position in averaged_positions.items():
#             rospy.loginfo(f'Averaged position for {artifact_labels[class_id]}: {position}')

#             # Find the 3D position based on the averaged position
#             artifact_distance = self.find_artifact_distance(position)

#             # Increment artifact ID counter if an artifact is found
#             if artifact_distance is not None:
#                 rospy.loginfo(f'Distance to {artifact_labels[class_id]}: {artifact_distance:.2f} meters')
           
#             # Check if it's a stop sign and approach it if not inspected
#             if class_id == 2:  # Assuming class_id for stop sign
#                 already_inspected = any(
#                     np.linalg.norm(np.array(position) - np.array(pos)) < self.inspection_threshold
#                     for pos in self.inspected_artifacts
#                 )
#                 if not already_inspected:
#                     self.artifact_found_ = True
#                     self.artifact_position_ = position  # Set the 3D artifact position
#                     rospy.loginfo('Self.artifact_position_ : {position}')
#                     self.approach_artifact(position)  # Pass the 3D point
#                     self.inspected_artifacts.append(position)

#         # Update artifact_found_ based on detections
#         self.artifact_found_ = len([d for d in detections if d[4] >= 0.75]) > 0

#         # Publish the image with bounding boxes for RVIZ display
#         image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="bgr8")
#         self.image_detections_pub_.publish(image_detection_message)

#         rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))


#     # Planning 2
#     def approach_artifact(self, position):
#         if position is None:
#             rospy.logwarn('Bounding box center (position) is None; skipping approach.')
#             return
        
#         # Calculate the goal position based on the artifact's position
#         u, v = position  # x and y in image pixels

#         rospy.loginfo('Approach_Artifact intiated. Position is {position}')

#         # Check if camera intrinsics are available
#         if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
#             rospy.loginfo('Camera intrinsics not available yet.')
#             return

#         closest_point = float('inf')
#         found_point = False
#         target_position = None

#         # Iterate over the original point cloud message to find the closest 3D point
#         for point in point_cloud2.read_points(self.current_pointcloud, skip_nans=True, field_names=("x", "y", "z")):
#             x, y, z = point
            
#             if z <= 0:  # Avoid division by zero
#                 continue

#             # Project 3D point to 2D
#             u_proj = int(self.fx * (x / z) + self.cx)
#             v_proj = int(self.fy * (y / z) + self.cy)

#             #rospy.loginfo(f'Checking point: {point} -> Projected: ({u_proj}, {v_proj}), Target: ({u}, {v})')

#             if abs(u_proj - u) < 5 and abs(v_proj - v) < 5:
#                 distance = np.sqrt(x**2 + y**2 + z**2)
#                 if distance < closest_point:
#                     closest_point = distance
#                     target_position = (x, y, z)
#                     found_point = True

#             if found_point:
#                 # Send a goal to "move_base" with "self.move_base_action_client_"
#                 action_goal = MoveBaseActionGoal()
#                 action_goal.goal.target_pose.header.frame_id = "map"
#                 action_goal.goal_id = self.goal_counter_
#                 self.goal_counter_ += 1

#                 pose_2d = Pose2D()
#                 pose_2d.x = target_position[0]
#                 pose_2d.y = target_position[1]
#                 pose_2d.theta = 0  # Adjust as needed

#                 action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)
#                 rospy.loginfo('Sending goal to approach artifact...')
#                 self.move_base_action_client_.send_goal(action_goal.goal)
#         else:
#             rospy.logwarn('No valid point found to approach artifact.')

 
#     # Planning
#     def exploration_strategy(self):
#         while not rospy.is_shutdown():
#             if self.artifact_found_:
#                 rospy.loginfo('Artifact detected, stopping exploration and approaching artifact.')
#                 self.approach_artifact(self.artifact_position_)
#                 self.inspected_artifacts.append(self.artifact_position_)
#                 self.artifact_found_ = False  # Reset the flag after inspection
#             else:
#                 # Continue with the existing exploration strategy
#                 self.planner_frontier_exploration(actionlib.GoalStatus.LOST)
#                 rospy.sleep(1)

#     def artifact_detection_callback(self, pose_msg):
#         artifact_id = pose_msg.header.seq  # Example identifier
#         if artifact_id not in self.inspected_artifacts_:
#             self.inspected_artifacts_.add(artifact_id)
#             self.inspection_mode_ = True
#             self.inspection_pose_ = pose_msg.pose
#             rospy.loginfo(f"Artifact detected: {artifact_id}. Switching to inspection mode.")

#     def planner_move_forwards(self, action_state):
#         # Simply move forward by 10m

#         # Only send this once before another action
#         if action_state == actionlib.GoalStatus.LOST:

#             pose_2d = self.get_pose_2d()

#             rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

#             # Move forward 10m
#             pose_2d.x += 10 * math.cos(pose_2d.theta)
#             pose_2d.y += 10 * math.sin(pose_2d.theta)

#             rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

#             # Send a goal to "move_base" with "self.move_base_action_client_"
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ = self.goal_counter_ + 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

#             rospy.loginfo('Sending goal...')
#             self.move_base_action_client_.send_goal(action_goal.goal)

#     def planner_go_to_first_artifact(self, action_state):
#         # Go to a pre-specified artifact (alien) location

#         # Only send this if not already going to a goal
#         if action_state != actionlib.GoalStatus.ACTIVE:

#             # Select a pre-specified goal location
#             pose_2d = Pose2D()
#             pose_2d.x = 18.0
#             pose_2d.y = 25.0
#             pose_2d.theta = -math.pi/2

#             # Send a goal to "move_base" with "self.move_base_action_client_"
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ = self.goal_counter_ + 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

#             rospy.loginfo('Sending goal...')
#             self.move_base_action_client_.send_goal(action_goal.goal)

#     def planner_return_home(self, action_state):
#         # Go to the origin

#         # Only send this if not already going to a goal
#         if action_state != actionlib.GoalStatus.ACTIVE:

#             # Select a pre-specified goal location
#             pose_2d = Pose2D()
#             pose_2d.x = 0
#             pose_2d.y = 0
#             pose_2d.theta = 0

#             # Send a goal to "move_base" with "self.move_base_action_client_"
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ = self.goal_counter_ + 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

#             rospy.loginfo('Sending goal...')
#             self.move_base_action_client_.send_goal(action_goal.goal)

#     def planner_random_walk(self, action_state):
#         # Go to a random location, which may be invalid

#         min_x = -5
#         max_x = 50
#         min_y = -5
#         max_y = 50

#         # Only send this if not already going to a goal
#         if action_state != actionlib.GoalStatus.ACTIVE:

#             # Select a random location
#             pose_2d = Pose2D()
#             pose_2d.x = random.uniform(min_x, max_x)
#             pose_2d.y = random.uniform(min_y, max_y)
#             pose_2d.theta = random.uniform(0, 2*math.pi)

#             # Send a goal to "move_base" with "self.move_base_action_client_"
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ = self.goal_counter_ + 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

#             rospy.loginfo('Sending goal...')
#             self.move_base_action_client_.send_goal(action_goal.goal)

#     def planner_random_goal(self, action_state):
#         # Go to a random location out of a predefined set

#         # Hand picked set of goal locations
#         random_goals = [[53.3,40.7],[44.4, 13.3],[2.3, 33.4],[9.9, 37.3],[3.4, 18.5],[6.0, 0.4],[28.3, 11.8],[43.7, 12.8],[38.9,43.0],[47.4,4.7],[31.5,3.2],[36.6,32.5]]

#         # Only send this if not already going to a goal
#         if action_state != actionlib.GoalStatus.ACTIVE:

#             # Select a random location
#             idx = random.randint(0,len(random_goals)-1)
#             pose_2d = Pose2D()
#             pose_2d.x = random_goals[idx][0]
#             pose_2d.y = random_goals[idx][1]
#             pose_2d.theta = random.uniform(0, 2*math.pi)

#             # Send a goal to "move_base" with "self.move_base_action_client_"
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ = self.goal_counter_ + 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

#             rospy.loginfo('Sending goal...')
#             self.move_base_action_client_.send_goal(action_goal.goal)

#     def find_frontiers(self):
#         with self.map_lock_:
#             if self.map_ is None:
#                 return []
#             grid = np.array(self.map_.data).reshape((self.map_.info.height, self.map_.info.width))
        
#         self.frontiers = []
#         for y in range(1, grid.shape[0]-1):
#             for x in range(1, grid.shape[1]-1):
#                 if grid[y, x] == -1:
#                     # Check if adjacent to free space
#                     if (grid[y+1, x] == 0 or grid[y-1, x] == 0 or
#                         grid[y, x+1] == 0 or grid[y, x-1] == 0):
#                         # Convert grid coordinates to world coordinates
#                         world_x = self.map_.info.origin.position.x + x * self.map_.info.resolution
#                         world_y = self.map_.info.origin.position.y + y * self.map_.info.resolution
#                         self.frontiers.append((world_x, world_y))
#         return self.frontiers
    
#     def planner_frontier_exploration(self, action_state):
#         if action_state != actionlib.GoalStatus.ACTIVE:
#             frontiers = self.find_frontiers()
#             if not frontiers:
#                 rospy.loginfo("No frontiers found. Exploration complete or stuck.")
#                 return
            
#             # Select the nearest frontier
#             current_pose = self.get_pose_2d()
#             min_dist = float('inf')
#             target = None
#             for world_x, world_y in frontiers:
#                 dist = math.sqrt((world_x - current_pose.x)**2 + (world_y - current_pose.y)**2)
#                 if dist < min_dist:
#                     min_dist = dist
#                     target = (world_x, world_y)
            
#             if target:
#                 pose_2d = Pose2D()
#                 pose_2d.x = target[0]
#                 pose_2d.y = target[1]
#                 pose_2d.theta = random.uniform(0, 2*math.pi)
                
#                 action_goal = MoveBaseActionGoal()
#                 action_goal.goal.target_pose.header.frame_id = "map"
#                 action_goal.goal_id = self.goal_counter_
#                 self.goal_counter_ += 1
#                 action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)
                
#                 rospy.loginfo(f"Sending frontier goal: {target}")
#                 self.move_base_action_client_.send_goal(action_goal.goal)
#             else:
#                 rospy.logwarn("No valid frontier found to send as a goal.")

#     def process_detections(self, bbox_centers):
#         class_positions = {}
#         class_counts = {}

#         # Loop through each bounding box center
#         for bbox_center, class_id in bbox_centers:
#             if class_id not in class_positions:
#                 class_positions[class_id] = np.array(bbox_center)
#                 class_counts[class_id] = 1
#             else:
#                 # Update average position incrementally
#                 class_positions[class_id] += np.array(bbox_center)
#                 class_counts[class_id] += 1

#         # Calculate the averaged positions
#         averaged_positions = {}
#         for class_id, total_position in class_positions.items():
#             averaged_positions[class_id] = (total_position / class_counts[class_id]).tolist()

#         return averaged_positions
             
#     def find_artifact_distance(self, position):
#         if position is None:  # Replace valid_conditions with your checks
#             rospy.logwarn('Invalid position returning None')
#             return None, None
        
#         u, v = position  # x and y in image pixels

#         # Check if camera intrinsics are available
#         if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
#             rospy.loginfo('Camera intrinsics not available yet.')
#             return None, None

#         closest_point = float('inf')
#         found_points = []
#         artifact_position = None

#         # Iterate over the original point cloud message to find the closest 3D point
#         for point in point_cloud2.read_points(self.current_pointcloud, skip_nans=True, field_names=("x", "y", "z")):
#             x, y, z = point
            
#             if z <= 0:  # Avoid division by zero
#                 continue

#             # Project 3D point to 2D
#             u_proj = int(self.fx * (x / z) + self.cx)
#             v_proj = int(self.fy * (y / z) + self.cy)

#             if abs(u_proj - u) < 5 and abs(v_proj - v) < 5:
#                 distance = np.sqrt(x**2 + y**2 + z**2)
#                 if distance < closest_point:
#                     closest_point = distance
#                     found_points.append((x, y, z))

#         if found_points:
#             # Calculate the average position
#             avg_x = sum(p[0] for p in found_points) / len(found_points)
#             avg_y = sum(p[1] for p in found_points) / len(found_points)
#             avg_z = sum(p[2] for p in found_points) / len(found_points)
            
#             artifact_position = (avg_x, avg_y, avg_z)

#             if artifact_position is not None:
#                 # Log the 3D artifact position
#                 rospy.loginfo(f'3D position for artifact: {artifact_position}')

#                 self.detected_artifacts.append(artifact_position)

#                 # Publish position
#                 self.publish_artifact_location(artifact_position)
#             else:
#                 rospy.loginfo('No found points in find_artifact_Disctance')

#         return closest_point

#     def planner_behavior_switching(self, action_state):
#         if self.state_ == ExplorationState.EXPLORING:
#             # Continue exploring
#             self.planner_frontier_exploration(action_state)
#             if self.artifact_found_:
#                 self.state_ = ExplorationState.INSPECTING
#         elif self.state_ == ExplorationState.INSPECTING:
#             self.planner_close_range_inspection(action_state)
#             if action_state == actionlib.GoalStatus.SUCCEEDED:
#                 self.state_ = ExplorationState.EXPLORING
#         elif self.state_ == ExplorationState.IDLE:
#             # Define idle behavior if necessary
#             pass

#     def planner_close_range_inspection(self, action_state):
#         if action_state != actionlib.GoalStatus.ACTIVE and self.inspection_mode_:
#             # Calculate inspection target (e.g., 1m in front of the artifact)
#             target_pose = Pose2D()
#             target_pose.x = self.inspection_pose_.position.x - 1.0 * math.cos(self.get_yaw_from_quaternion(self.inspection_pose_.orientation))
#             target_pose.y = self.inspection_pose_.position.y - 1.0 * math.sin(self.get_yaw_from_quaternion(self.inspection_pose_.orientation))
#             target_pose.theta = self.get_yaw_from_quaternion(self.inspection_pose_.orientation)
           
#             action_goal = MoveBaseActionGoal()
#             action_goal.goal.target_pose.header.frame_id = "map"
#             action_goal.goal_id = self.goal_counter_
#             self.goal_counter_ += 1
#             action_goal.goal.target_pose.pose = pose2d_to_pose(target_pose)
           
#             rospy.loginfo(f"Sending inspection goal: ({target_pose.x}, {target_pose.y})")
#             self.move_base_action_client_.send_goal(action_goal.goal)
#             self.inspection_mode_ = False  # Reset after sending goal
   
#     def get_yaw_from_quaternion(self, orientation):
#         euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
#         return euler[2]
    
#     def publish_artifact_location(self, artifact_position):
#         for i, pos in enumerate(self.detected_artifacts):
#             # Create a marker
#             marker = Marker()
#             marker.header.frame_id = "map"  # Set the frame
#             marker.id = 0
#             marker.type = Marker.SPHERE 
#             marker.action = Marker.ADD
#             marker.pose.position.x = artifact_position[0]
#             marker.pose.position.y = artifact_position[1]
#             marker.pose.position.z = artifact_position[2]
#             marker.scale.x = 1.0  # Size of marker
#             marker.scale.y = 1.0
#             marker.scale.z = 1.0
#             marker.color.a = 1.0  # Transparency
#             marker.color.r = 1.0  # Red color
#             marker.color.g = 0.0
#             marker.color.b = 0.0

#             # Publish the marker
#             self.artifact_marker_pub_.publish(marker)
#             rospy.loginfo(f"Published artifact marker at {artifact_position}.")


#     def main_loop(self):
        
#         while not rospy.is_shutdown():

#             action_state = self.move_base_action_client_.get_state()
#             rospy.loginfo(f"Action State: {self.move_base_action_client_.get_goal_status_text()}")
           
#             if self.planner_type_ == PlannerType.CLOSE_RANGE_INSPECTION:
#                 self.planner_close_range_inspection(action_state)
#             # Existing planner selection logic...
           
#             rospy.sleep(0.5)

#             #######################################################
#             # Get the current status
#             # See the possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
#             action_state = self.move_base_action_client_.get_state()
#             rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
#             #rospy.loginfo('action_state number:' + str(action_state))

#             if action_state == actionlib.GoalStatus.ACTIVE:
#                 # The goal is currently being executed
#                 continue
#             if action_state == actionlib.GoalStatus.PENDING:
#                 rospy.loginfo("Goal is pending...")
#             if action_state == actionlib.GoalStatus.PREEMPTED:
#                 rospy.loginfo("Goal was PREEMPTED.")
#             if action_state == actionlib.GoalStatus.ABORTED:
#                 rospy.loginfo("Goal was ABORTED.")
#             if action_state == actionlib.GoalStatus.REJECTED:
#                 rospy.loginfo("Goal was REJECTED.")
#             if action_state == actionlib.GoalStatus.PREEMPTING:
#                 rospy.loginfo("Goal is PREEMPTING.")
#             if action_state == actionlib.GoalStatus.RECALLING:
#                 rospy.loginfo("Goal is RECALLING.")
#             if action_state == actionlib.GoalStatus.RECALLED:
#                 rospy.loginfo("Goal was RECALLED")
#             if action_state == actionlib.GoalStatus.LOST:
#                 rospy.loginfo("Goal is LOST.")

#             if (self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT) and (action_state == actionlib.GoalStatus.SUCCEEDED):
#                 print("Successfully reached first artifact!")
#                 self.reached_first_artifact_ = True
#             if (self.planner_type_ == PlannerType.RETURN_HOME) and (action_state == actionlib.GoalStatus.SUCCEEDED):
#                 print("Successfully returned home!")
#                 self.returned_home_ = True
#             if (self.planner_type_ == PlannerType.FRONTIER_EXPLORATION) and (action_state == actionlib.GoalStatus.SUCCEEDED):
#                 print("Successfully reached frontier goal!")
#                 self.reached_frontier_goal_ = True


#             #######################################################
#             # Select the next planner to execute
#             # Update this logic as you see fit!
#             self.planner_type_ = PlannerType.MOVE_FORWARDS

#             if self.planner_type_ != PlannerType.FRONTIER_EXPLORATION and not self.reached_frontier_goal_:
#                 self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
#             if not self.reached_first_artifact_ and self.planner_type_ != PlannerType.FRONTIER_EXPLORATION:
#                 self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
#             elif not self.returned_home_:
#                 self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
#             else:
#                 self.planner_type_ = PlannerType.FRONTIER_EXPLORATION


#             #######################################################
#             # Execute the planner by calling the relevant method
#             # The methods send a goal to "move_base" with "self.move_base_action_client_"
#             # Add your own planners here!
#             print("Calling planner:", self.planner_type_.name)
#             if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
#                 self.planner_frontier_exploration(action_state)
#             elif self.planner_type_ == PlannerType.MOVE_FORWARDS:
#                 self.planner_move_forwards(action_state)
#             elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
#                 self.planner_go_to_first_artifact(action_state)
#             elif self.planner_type_ == PlannerType.RETURN_HOME:
#                 self.planner_return_home(action_state)
#             elif self.planner_type_ == PlannerType.RANDOM_WALK:
#                 self.planner_random_walk(action_state)
#             elif self.planner_type_ == PlannerType.RANDOM_GOAL:
#                 self.planner_random_goal(action_state)


#             #######################################################
#             # Delay so the loop doesn't run too fast
#             rospy.sleep(0.2)



# if __name__ == '__main__':

#     # Create the ROS node
#     rospy.init_node('cave_explorer')

#     # Create the cave explorer
#     cave_explorer = CaveExplorer()
        
#     # Start the exploration strategy
#     #cave_explorer.exploration_strategy()

#     # Loop forever while processing callbacks
#     cave_explorer.main_loop()




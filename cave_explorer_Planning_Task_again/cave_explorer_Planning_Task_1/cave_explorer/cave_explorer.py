#!/usr/bin/env python3

import math
import random
from enum import Enum
################################
import os
import datetime
import time
import numpy as np

# For trained model 
try:
    import torch  # For running the trained YOLOv5 model
except ImportError:
    torch = None
################################

import cv2  # OpenCV2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
################################
from sensor_msgs.msg import Image
################################
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORATION = 6
    # Add more!


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False
        

        ################################
        # TASK 1: DATASET COLLECTION VARIABLES
        ################################
        self.save_interval = 3.0  # Save image every 3 seconds
        self.last_saved_time = 0.0
        self.image_counter = 0
        
        ################################
        # TASK 2: ARTIFACT DETECTION VARIABLES & MODEL LOADING
        ################################
        self.custom_model = None
        try:
            import torch
            self.get_logger().info("Attempting to import torch...")
            self.get_logger().info(f"✅ PyTorch imported successfully! Version: {torch.__version__}")
            
            model_path = '/home/student/ros_ws/CV_Model/yolov5/best.pt'
            repo_path = '/home/student/ros_ws/CV_Model/yolov5'
            
            # Check if files exist
            import os
            self.get_logger().info(f"Model file exists: {os.path.exists(model_path)}")
            self.get_logger().info(f"Repo path exists: {os.path.exists(repo_path)}")
            
            if not os.path.exists(model_path):
                self.get_logger().error(f"❌ Model file not found at: {model_path}")
                self.use_custom_model = False
            elif not os.path.exists(repo_path):
                self.get_logger().error(f"❌ YOLOv5 repo not found at: {repo_path}")
                self.use_custom_model = False
            else:
                self.get_logger().info("Attempting to load YOLOv5 model...")
                try:
                    # PyTorch 2.9+ requires weights_only=False for YOLOv5 models
                    # This is safe since we trust our own trained model
                    self.get_logger().info("Loading model with PyTorch 2.9+ compatibility...")
                    self.custom_model = torch.load(model_path, map_location='cpu', weights_only=False)
                    
                    # Set model to evaluation mode
                    self.custom_model.eval()
                    
                    self.get_logger().info("✅ Successfully loaded custom trained YOLOv5 model!")
                    self.use_custom_model = True
                    
                except Exception as load_error:
                    self.get_logger().error(f"Direct model loading failed: {load_error}")
                    
                    # Fallback: Try with YOLOv5 repo structure
                    try:
                        import sys
                        original_cwd = os.getcwd()
                        os.chdir(repo_path)
                        sys.path.insert(0, repo_path)
                        
                        # Try torch.hub method from YOLOv5 directory
                        self.custom_model = torch.hub.load('.', 'custom', path='best.pt', source='local', trust_repo=True)
                        self.get_logger().info("✅ Successfully loaded via torch.hub from YOLOv5 directory!")
                        self.use_custom_model = True
                        
                        os.chdir(original_cwd)
                        
                    except Exception as hub_error:
                        self.get_logger().error(f"Torch.hub loading also failed: {hub_error}")
                        
                        # Final fallback: Skip model loading but continue with image collection
                        self.get_logger().warn("Could not load trained model - continuing in data collection mode")
                        self.use_custom_model = False
            
        except ImportError as e:
            self.get_logger().error(f"❌ Could not import PyTorch: {e}")
            self.get_logger().error("PyTorch is not installed in the current Python environment")
            self.use_custom_model = False
            
        except Exception as e:
            self.get_logger().error(f"❌ Could not load custom model: {e}")
            self.get_logger().error("Falling back to no detection mode.")
            self.use_custom_model = False
        
        ################################
        # TASK 3: ARTIFACT LOCALIZATION VARIABLES
        ################################
        self.detected_artifacts_positions = []  # Store 3D positions
        self.artifact_clustering_threshold = 4.5  # Very aggressive clustering to prevent duplicates
        ################################

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        
        ################################
        # PLANNING TASK 1: FRONTIER EXPLORATION VARIABLES
        ################################
        self.current_map_ = None
        self.map_info_ = None
        self.exploration_complete_ = False
        self.min_frontier_size_ = 5  # Minimum frontier cluster size
        self.exploration_radius_ = 3.0  # Minimum distance between goals
        ################################

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        ################################
        # PLANNING TASK 1: FRONTIER EXPLORATION PUBLISHERS
        ################################
        self.frontier_marker_pub_ = self.create_publisher(MarkerArray, 'frontier_markers', 10)
        self.goal_marker_pub_ = self.create_publisher(Marker, 'current_goal_marker', 10)
        ################################

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)
        
        ################################
        # Task 3: Simple approach - no complex sensor subscriptions needed
        # We'll use the simple geometric approach for localization
        ################################

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits and store map data for frontier detection"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        ################################
        # PLANNING TASK 1: STORE MAP DATA FOR FRONTIER EXPLORATION
        ################################
        self.current_map_ = np.array(map_msg.data).reshape((map_height, map_width))
        self.map_info_ = map_msg.info
        ################################

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
    
    def image_callback(self, image_msg):
        """
        Enhanced image callback implementing all three perception tasks:
        1. Dataset collection 
        2. Advanced artifact detection
        3. Artifact localization
        """
    
        # Copy the image message to a cv image
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        
        ################################
        # Make sure image is writable
        if not image.flags.writeable:
            image = image.copy()

        ################################
        # TASK 1: DATASET COLLECTION
        ################################
        self.save_images_for_dataset(image)
        
        ################################
        # TASK 2: ARTIFACT DETECTION
        ################################
        detections = self.detect_artifacts_advanced(image)
        
        # Debug logging
        if len(detections) > 0:
            self.get_logger().info(f'Raw detections: {len(detections)}')
            for detection in detections:
                _, _, _, _, confidence, artifact_type = detection
                self.get_logger().info(f'  - {artifact_type}: {confidence:.2f}')
        
        # Filter detections by confidence threshold - use adaptive thresholds per type
        filtered_detections = []
        for detection in detections:
            x, y, w, h, confidence, artifact_type = detection
            
            # Type-specific confidence thresholds to balance detection vs false positives
            type_thresholds = {
                'stop_sign': 0.7,        # Keep high confidence for stop signs
                'white_sphere': 0.3,     # Lower threshold for white spheres
                'green_crystal': 0.15,   # Very low threshold
                'green_alien': 0.15,     # Very low threshold
                'mushrooms': 0.15,       # Very low threshold
                'formation': 0.7         # Much higher threshold to reduce wall confusion
            }
            
            threshold = type_thresholds.get(artifact_type, 0.4)
            if confidence >= threshold:
                filtered_detections.append(detection)
        
        # Debug logging for filtered detections
        if len(filtered_detections) != len(detections):
            self.get_logger().info(f'After filtering: {len(filtered_detections)} (removed {len(detections) - len(filtered_detections)})')
        
        ################################
        # TASK 3: ARTIFACT LOCALIZATION
        ################################
        artifact_3d_positions = []
        for detection in filtered_detections:
            x, y, width, height, confidence, artifact_type = detection
            center_x = x + width // 2
            center_y = y + height // 2
            
            self.get_logger().info(f'Localizing {artifact_type} at pixel ({center_x}, {center_y})')
            
            # Get 3D position of this detection
            artifact_3d_pos = self.get_artifact_3d_position(center_x, center_y)
            if artifact_3d_pos is not None:
                artifact_3d_positions.append((artifact_3d_pos, artifact_type, confidence))
                self.get_logger().info(f'Successfully localized {artifact_type} at 3D position: ({artifact_3d_pos[0]:.2f}, {artifact_3d_pos[1]:.2f}, {artifact_3d_pos[2]:.2f})')
            else:
                self.get_logger().warn(f'Failed to get 3D position for {artifact_type} at pixel ({center_x}, {center_y})')
        
        # Update artifact_found_ flag based on filtered detections
        self.artifact_found_ = len(filtered_detections) > 0
        
        # Store detected artifact positions for clustering/averaging
        for pos_data in artifact_3d_positions:
            pos_3d, artifact_type, confidence = pos_data
            self.add_detected_artifact(pos_3d, artifact_type, confidence)

        ################################
        # VISUALIZATION & OUTPUT
        ################################
        # Draw bounding boxes on image for visualization (only for filtered detections)
        for detection in filtered_detections:
            x, y, width, height, confidence, artifact_type = detection
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Add label with artifact type and confidence
            label = f"{artifact_type}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish the annotated image
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        if self.artifact_found_:
            self.get_logger().info(f'Artifact found! {len(filtered_detections)} detections')
            for detection in filtered_detections:
                _, _, _, _, confidence, artifact_type = detection
                self.get_logger().info(f'  - {artifact_type}: {confidence:.2f}')
            self.localise_artifact()

    ################################
    # TASK 1: DATASET COLLECTION METHODS
    ################################
    
    def save_images_for_dataset(self, image):
        """
        TASK 1: Save images periodically for building training dataset
        These images are used to train the YOLOv5 model used in Task 2 
        
        WORKFLOW:
        1. This method saves raw images during exploration
        2. Images are manually labeled (using tools like LabelImg or Roboflow)
        3. YOLOv5 model is trained on labeled dataset
        4. Trained model (best.pt) is used in Task 2 for detection
        """
        current_time = time.time()
        
        # Only save if enough time has passed since last save
        if current_time - self.last_saved_time >= self.save_interval:
            # Create save directory 
            save_directory = "/home/student/ros_ws/dataset_images"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cave_image_{self.image_counter:04d}_{timestamp}.jpg"
            filepath = os.path.join(save_directory, filename)
            
            # Save the image
            success = cv2.imwrite(filepath, image)
            if success:
                self.image_counter += 1
                self.last_saved_time = current_time
                self.get_logger().info(f'Saved image {self.image_counter}: {filename}')
            else:
                self.get_logger().warn(f'Failed to save image: {filename}')

    ################################
    # TASK 2: ARTIFACT DETECTION METHODS
    ################################

    def detect_artifacts_advanced(self, image):
        """
        TASK 2: Artifact detection using trained YOLOv5 model
        Uses trained model from Task 1 dataset
        Returns: List of detections (x, y, width, height, confidence, artifact_type)
        """
        detections = []
        
        self.get_logger().debug('Starting artifact detection with trained model...')
        
        # Use custom trained model 
        if self.use_custom_model and self.custom_model is not None:
            detections = self.detect_with_trained_model(image)
            self.get_logger().info(f'Trained model detections: {len(detections)}')
        else:
            self.get_logger().warn('No trained model available - no detections possible')
        
        return detections
    
    def detect_with_trained_model(self, image):
        """
        Use trained YOLOv5 model (trained on Task 1 dataset) for artifact detection
        Returns: List of detections (x, y, width, height, confidence, artifact_type)
        """
        detections = []
        
        try:
            # Run YOLOv5 model on the image
            results = self.custom_model(image)
            model_detections = results.xyxy[0].cpu().numpy()
            
            # Artifact labels matching the training data
            artifact_labels = ['green_crystal', 'green_alien', 'stop_sign', 'mushrooms', 'formation', 'white_sphere']
            
            # Process detections 
            for detection in model_detections:
                x1, y1, x2, y2, conf, cls = detection
                
                # Apply lower thresholds initially to get more detections
                if conf >= 0.1:  # Very low threshold initially
                    class_id = int(cls)
                    if class_id < len(artifact_labels):
                        artifact_type = artifact_labels[class_id]
                        
                        # Convert to our format (x, y, width, height, confidence, type)
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        
                        detections.append((x, y, w, h, float(conf), artifact_type))
                        
                        self.get_logger().info(f'Model detected {artifact_type} with confidence {conf:.2f}')
            
            self.get_logger().info(f'Trained model found {len(detections)} detections')
            
        except Exception as e:
            self.get_logger().error(f'Error using trained model: {e}')
            return []
        
        return detections
    
    ################################
    # TASK 3: ARTIFACT LOCALIZATION METHODS
    ################################

    def get_artifact_3d_position(self, pixel_x, pixel_y):
        """
        TASK 3: Get 3D world position of artifact from 2D pixel coordinates
        Simple approach: Use pixel location to estimate position in front of robot
        """
        if pixel_x < 0 or pixel_y < 0:
            return None
        
        # Get current robot pose
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return None
        
        # Simple approach: Place artifact at reasonable distance in direction robot is looking
        # Use pixel position to determine left/right offset from robot's center view
        
        # Assume image center corresponds to robot's forward direction
        image_width = 640  # Typical camera width
        image_center_x = image_width / 2
        
        # Calculate horizontal offset from center (positive = right, negative = left)
        pixel_offset_x = pixel_x - image_center_x
        
        # Convert pixel offset to angular offset (simple approximation)
        # Assume 60 degree field of view, so full width = 60 degrees
        fov_degrees = 60.0
        angular_offset = (pixel_offset_x / image_width) * math.radians(fov_degrees)
        
        # Calculate artifact's direction relative to robot
        artifact_direction = robot_pose.theta + angular_offset
        
        # Place artifact at reasonable distance (3-5 meters in front of robot)
        artifact_distance = 4.0  # Fixed distance for simplicity
        
        # Calculate world coordinates
        artifact_x = robot_pose.x + artifact_distance * math.cos(artifact_direction)
        artifact_y = robot_pose.y + artifact_distance * math.sin(artifact_direction)
        artifact_z = 0.5  # Reasonable height above ground
        
        self.get_logger().info(f'Artifact at pixel ({pixel_x}, {pixel_y}) -> world ({artifact_x:.2f}, {artifact_y:.2f}, {artifact_z:.2f})')
        self.get_logger().info(f'  Robot pose: ({robot_pose.x:.2f}, {robot_pose.y:.2f}, {robot_pose.theta:.2f})')
        self.get_logger().info(f'  Angular offset: {math.degrees(angular_offset):.1f} degrees')
        
        return (artifact_x, artifact_y, artifact_z)
    
    # All complex sensor processing methods removed - using simple geometric approach
        
    def add_detected_artifact(self, position_3d, artifact_type, confidence):
        """
        Add detected artifact with clustering to avoid duplicates and improve accuracy
        """
        if position_3d is None:
            return
        
        # Safety checks to prevent false artifact positioning
        robot_pose = self.get_pose_2d()
        if robot_pose is not None:
            robot_distance = math.sqrt(
                (position_3d[0] - robot_pose.x)**2 + 
                (position_3d[1] - robot_pose.y)**2
            )
            
            # Reject artifacts too close to robot (estimation errors)
            if robot_distance < 1.0:  # Increased to 1.0m for safety
                self.get_logger().warn(f'Rejecting {artifact_type} too close to robot ({robot_distance:.2f}m) - likely estimation error')
                return
            
            # Reject artifacts too far from robot (outliers)
            if robot_distance > 12.0:  # Maximum reasonable detection distance
                self.get_logger().warn(f'Rejecting {artifact_type} too far from robot ({robot_distance:.2f}m) - likely outlier')
                return
        
        # Reject artifacts with unreasonable heights
        if position_3d[2] < -1.0 or position_3d[2] > 5.0:  # Reasonable height range
            self.get_logger().warn(f'Rejecting {artifact_type} with unreasonable height ({position_3d[2]:.2f}m)')
            return
            
        # Check if this detection is close to an existing one (clustering)
        # Use more aggressive clustering to prevent duplicate markers
        closest_match = None
        closest_distance = float('inf')
        
        for i, (existing_pos, existing_type, existing_conf) in enumerate(self.detected_artifacts_positions):
            # Cluster same types, but also consider if different types are very close (might be misclassification)
            same_type = existing_type == artifact_type
            very_close = False
            
            distance = math.sqrt(
                (position_3d[0] - existing_pos[0])**2 + 
                (position_3d[1] - existing_pos[1])**2 + 
                (position_3d[2] - existing_pos[2])**2
            )
            
            # For same artifact type, use normal clustering threshold
            if same_type and distance < self.artifact_clustering_threshold:
                if distance < closest_distance:
                    closest_match = i
                    closest_distance = distance
            
            # For different types but very close (< 2.5m), also cluster to prevent duplicates
            elif not same_type and distance < 2.5:
                if distance < closest_distance:
                    closest_match = i
                    closest_distance = distance
                    very_close = True
        
        if closest_match is not None:
            # Update existing artifact with averaged position and higher confidence
            existing_pos, existing_type, existing_conf = self.detected_artifacts_positions[closest_match]
            
            # For very close but different types, keep the type with higher confidence
            if very_close and confidence > existing_conf:
                final_type = artifact_type
                self.get_logger().info(f'Updating artifact type from {existing_type} to {artifact_type} due to higher confidence')
            else:
                final_type = existing_type
            
            # Weight by confidence for better averaging
            total_conf = existing_conf + confidence
            weight_existing = existing_conf / total_conf
            weight_new = confidence / total_conf
            
            # Weighted average of positions
            avg_x = existing_pos[0] * weight_existing + position_3d[0] * weight_new
            avg_y = existing_pos[1] * weight_existing + position_3d[1] * weight_new
            avg_z = existing_pos[2] * weight_existing + position_3d[2] * weight_new
            
            # Update the stored artifact
            self.detected_artifacts_positions[closest_match] = ((avg_x, avg_y, avg_z), final_type, total_conf)
            
            # Update corresponding marker point
            self.artifact_locations_[closest_match].x = avg_x
            self.artifact_locations_[closest_match].y = avg_y
            self.artifact_locations_[closest_match].z = avg_z
            
            self.get_logger().info(f'Clustered: Updated {final_type} position to ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}) with combined confidence {total_conf:.2f}')
        else:
            # Add new artifact if not a duplicate        
            self.detected_artifacts_positions.append((position_3d, artifact_type, confidence))
            self.get_logger().info(f'New unique {artifact_type} detected at 3D position: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}) with confidence {confidence:.2f}')
            
            # Add to artifact locations for marker display
            point = Point()
            point.x = position_3d[0]  
            point.y = position_3d[1]
            point.z = position_3d[2]
            self.artifact_locations_.append(point)
        
        # After adding/updating, perform cleanup to merge any remaining close artifacts
        self.cleanup_close_artifacts()
        
        # Publish updated markers
        self.publish_artifact_markers()
        
    def cleanup_close_artifacts(self):
        """
        Post-processing cleanup to merge artifacts that are still too close together
        This helps eliminate any remaining duplicate markers
        """
        if len(self.detected_artifacts_positions) < 2:
            return
            
        # Very aggressive final cleanup - merge anything within 2.0m
        cleanup_threshold = 2.0
        
        merged_indices = set()
        for i in range(len(self.detected_artifacts_positions)):
            if i in merged_indices:
                continue
                
            pos_i, type_i, conf_i = self.detected_artifacts_positions[i]
            
            # Find all other artifacts within cleanup threshold
            to_merge = [i]
            for j in range(i + 1, len(self.detected_artifacts_positions)):
                if j in merged_indices:
                    continue
                    
                pos_j, type_j, conf_j = self.detected_artifacts_positions[j]
                distance = math.sqrt(
                    (pos_i[0] - pos_j[0])**2 + 
                    (pos_i[1] - pos_j[1])**2 + 
                    (pos_i[2] - pos_j[2])**2
                )
                
                if distance < cleanup_threshold:
                    to_merge.append(j)
                    merged_indices.add(j)
            
            # If we found artifacts to merge
            if len(to_merge) > 1:
                self.get_logger().info(f'Final cleanup: Merging {len(to_merge)} close artifacts')
                
                # Calculate weighted average of all positions and confidences
                total_conf = 0
                weighted_x, weighted_y, weighted_z = 0, 0, 0
                best_type = type_i
                best_conf = conf_i
                
                for idx in to_merge:
                    pos, artifact_type, conf = self.detected_artifacts_positions[idx]
                    total_conf += conf
                    weighted_x += pos[0] * conf
                    weighted_y += pos[1] * conf  
                    weighted_z += pos[2] * conf
                    
                    # Keep the type with highest confidence
                    if conf > best_conf:
                        best_type = artifact_type
                        best_conf = conf
                
                # Calculate final averaged position
                if total_conf > 0:
                    final_x = weighted_x / total_conf
                    final_y = weighted_y / total_conf
                    final_z = weighted_z / total_conf
                    
                    # Update the first artifact with merged data
                    self.detected_artifacts_positions[i] = ((final_x, final_y, final_z), best_type, total_conf)
                    self.artifact_locations_[i].x = final_x
                    self.artifact_locations_[i].y = final_y
                    self.artifact_locations_[i].z = final_z
        
        # Remove merged artifacts (in reverse order to maintain indices)
        if merged_indices:
            for idx in sorted(merged_indices, reverse=True):
                del self.detected_artifacts_positions[idx]
                del self.artifact_locations_[idx]
            
            self.get_logger().info(f'Final cleanup: Removed {len(merged_indices)} duplicate artifacts')

    ################################
    # END OF PERCEPTION TASKS - REMAINING METHODS ARE FOR SYSTEM SUPPORT
    ################################


    def localise_artifact(self):
        """
        TASK 3: Compute the location of detected artifacts using camera data
        This now uses the actual 3D positions computed from point cloud data
        rather than just the robot location
        """
        
        # Get current robot pose for transformation if needed
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            self.get_logger().warn('localise_artifact: robot_pose is None.')
            return

        # The artifact locations are now already computed in 3D and stored
        # in self.detected_artifacts_positions by the image_callback method
        # The artifact_locations_ list is updated in add_detected_artifact()
        
        # Log the current number of detected artifacts
        num_artifacts = len(self.detected_artifacts_positions)
        if num_artifacts > 0:
            self.get_logger().info(f'Total artifacts localized: {num_artifacts}')
            
            # Log the most recent detection
            latest_pos, latest_type, latest_conf = self.detected_artifacts_positions[-1]
            self.get_logger().info(
                f'Latest: {latest_type} at ({latest_pos[0]:.2f}, {latest_pos[1]:.2f}, {latest_pos[2]:.2f}) '
                f'with confidence {latest_conf:.2f}'
            )
        
        # Markers are already published in add_detected_artifact()
        
        # Log current artifact status - only show first 10 to avoid spam
        if len(self.detected_artifacts_positions) > 0:
            self.get_logger().info('=== Current Detected Artifacts (First 10) ===')
            for i, (pos, artifact_type, conf) in enumerate(self.detected_artifacts_positions[:10]):
                self.get_logger().info(f'{i+1}. {artifact_type}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) conf={conf:.2f}')
            if len(self.detected_artifacts_positions) > 10:
                self.get_logger().info(f'... and {len(self.detected_artifacts_positions) - 10} more artifacts')
            self.get_logger().info('=== End Artifact List ===')
        
    def get_perception_statistics(self):
        """
        Get statistics about the perception system performance
        Useful for debugging and performance evaluation
        """
        stats = {
            'total_images_saved': self.image_counter,
            'total_artifacts_detected': len(self.detected_artifacts_positions),
            'artifacts_by_type': {}
        }
        
        # Count artifacts by type
        for _, artifact_type, _ in self.detected_artifacts_positions:
            if artifact_type not in stats['artifacts_by_type']:
                stats['artifacts_by_type'][artifact_type] = 0
            stats['artifacts_by_type'][artifact_type] += 1
            
        return stats
        
    def log_perception_status(self):
        """Log current perception system status"""
        stats = self.get_perception_statistics()
        
        self.get_logger().info('=== Perception System Status ===')
        self.get_logger().info(f'Images collected: {stats["total_images_saved"]}')
        self.get_logger().info(f'Artifacts detected: {stats["total_artifacts_detected"]}')
        
        if stats['artifacts_by_type']:
            self.get_logger().info('Artifacts by type:')
            for artifact_type, count in stats['artifacts_by_type'].items():
                self.get_logger().info(f'  {artifact_type}: {count}')
        
        self.get_logger().info('=== End Status ===')

    def publish_artifact_markers(self):
        """
        Publish enhanced artifact location markers with different colors for different types
        """
        marker_array = MarkerArray()
        
        # Create a marker for each detected artifact type
        artifact_type_colors = {
            'stop_sign': (1.0, 0.0, 0.0),      # Red
            'green_crystal': (0.0, 1.0, 0.0),  # Green  
            'green_alien': (0.0, 0.8, 0.2),    # Dark green
            'white_sphere': (1.0, 1.0, 1.0),   # White
            'mushrooms': (0.8, 0.4, 0.0),      # Orange
            'formation': (0.5, 0.5, 0.8)       # Blue
        }
        
        # Group artifacts by type
        artifacts_by_type = {}
        for i, (pos_3d, artifact_type, confidence) in enumerate(self.detected_artifacts_positions):
            if artifact_type not in artifacts_by_type:
                artifacts_by_type[artifact_type] = []
            artifacts_by_type[artifact_type].append(pos_3d)
        
        # Create a separate marker for each artifact type
        marker_id = 0
        for artifact_type, positions in artifacts_by_type.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"artifacts_{artifact_type}"
            marker.id = marker_id
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            
            # Set marker pose
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Set marker scale - make them bigger and more visible
            marker.scale.x = 1.5
            marker.scale.y = 1.5
            marker.scale.z = 1.5
            
            # Set color based on artifact type
            color = artifact_type_colors.get(artifact_type, (1.0, 0.0, 1.0))  # Default magenta
            marker.color.r = color[0]
            marker.color.g = color[1] 
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            # Add all positions of this type
            for pos_3d in positions:
                point = Point()
                point.x = pos_3d[0]
                point.y = pos_3d[1] 
                point.z = pos_3d[2]
                marker.points.append(point)
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Also keep the original single marker for backward compatibility
        if self.artifact_locations_:
            self.marker_artifacts_.points = self.artifact_locations_
            marker_array.markers.append(self.marker_artifacts_)
        
        # Publish the marker array
        self.marker_pub_.publish(marker_array)

##############################################################################################################################3
    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True


    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    ################################
    # PLANNING TASK 1: FRONTIER EXPLORATION METHODS
    ################################
    
    def planner_frontier_exploration(self):
        """Efficient WFD-based frontier exploration using queue for BFS"""
        if self.current_map_ is None or self.map_info_ is None:
            return False
        
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return False
        
        # Find frontier centroids using WFD algorithm
        frontiers = self.wfd_frontier_detection(robot_pose)
        
        if not frontiers:
            self.get_logger().info('No frontiers found - exploration complete!')
            self.exploration_complete_ = True
            return False
        
        # Select closest valid frontier
        best_frontier = min(frontiers, key=lambda f: 
            math.sqrt((f[0] - robot_pose.x)**2 + (f[1] - robot_pose.y)**2))
        
        # Navigate to frontier
        goal_pose2d = Pose2D(x=best_frontier[0], y=best_frontier[1], theta=0.0)
        self.visualize_exploration(frontiers, best_frontier)
        
        self.get_logger().info(f'Exploring frontier at ({best_frontier[0]:.1f}, {best_frontier[1]:.1f})')
        self.planner_go_to_pose2d(goal_pose2d)
        return True
    
    def wfd_frontier_detection(self, robot_pose):
        """Wavefront Frontier Detection - efficient clustering approach"""
        if self.current_map_ is None:
            return []
        
        height, width = self.current_map_.shape
        marked = np.zeros((height, width), dtype=bool)
        frontiers = []
        
        # Queue for BFS: (y, x)  
        queue = []
        
        # Convert robot position to grid coordinates
        robot_x = int((robot_pose.x - self.map_info_.origin.position.x) / self.map_info_.resolution)
        robot_y = int((robot_pose.y - self.map_info_.origin.position.y) / self.map_info_.resolution)
        
        # Start BFS from robot position if valid
        if 0 <= robot_x < width and 0 <= robot_y < height:
            queue.append((robot_y, robot_x))
            marked[robot_y, robot_x] = True
        
        # BFS to find all reachable frontiers
        while queue:
            y, x = queue.pop(0)  # DEQUEUE
            
            if self.is_frontier_point(y, x):
                # Find connected frontier cluster
                new_frontier = self.extract_frontier_cluster(y, x, marked)
                if len(new_frontier) >= self.min_frontier_size_:
                    # Get centroid of cluster
                    centroid_x = sum(p[1] for p in new_frontier) / len(new_frontier)
                    centroid_y = sum(p[0] for p in new_frontier) / len(new_frontier)
                    
                    # Convert to world coordinates
                    world_x = self.map_info_.origin.position.x + centroid_x * self.map_info_.resolution
                    world_y = self.map_info_.origin.position.y + centroid_y * self.map_info_.resolution
                    
                    # Check if far enough from robot (avoid revisiting)
                    distance = math.sqrt((world_x - robot_pose.x)**2 + (world_y - robot_pose.y)**2)
                    if distance > self.exploration_radius_:
                        frontiers.append((world_x, world_y))
            
            # Add adjacent free cells to queue
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and 
                    not marked[ny, nx] and self.current_map_[ny, nx] == 0):
                    queue.append((ny, nx))  # ENQUEUE
                    marked[ny, nx] = True
        
        return frontiers
    
    def is_frontier_point(self, y, x):
        """Check if cell is frontier (free space adjacent to unknown)"""
        if self.current_map_[y, x] != 0:  # Must be free space
            return False
        
        height, width = self.current_map_.shape
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if self.current_map_[ny, nx] == -1:  # Adjacent to unknown
                    return True
        return False
    
    def extract_frontier_cluster(self, start_y, start_x, marked):
        """Extract connected frontier cluster using BFS"""
        cluster = []
        queue = [(start_y, start_x)]
        height, width = self.current_map_.shape
        
        while queue:
            y, x = queue.pop(0)
            if marked[y, x]:
                continue
                
            marked[y, x] = True
            cluster.append((y, x))
            
            # Add adjacent frontier points
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and 
                    not marked[ny, nx] and self.is_frontier_point(ny, nx)):
                    queue.append((ny, nx))
        
        return cluster
    
    def visualize_exploration(self, frontiers, selected_frontier):
        """Visualize frontiers with distinct colors from artifact markers"""
        marker_array = MarkerArray()
        
        # Frontier points as cyan cubes (different from artifact spheres)
        if frontiers:
            frontier_marker = Marker()
            frontier_marker.header.frame_id = "map"
            frontier_marker.header.stamp = self.get_clock().now().to_msg()
            frontier_marker.ns = "exploration_frontiers" 
            frontier_marker.id = 0
            frontier_marker.type = Marker.CUBE_LIST  # Different shape
            frontier_marker.action = Marker.ADD
            
            frontier_marker.pose.orientation.w = 1.0
            frontier_marker.scale.x = 0.3
            frontier_marker.scale.y = 0.3  
            frontier_marker.scale.z = 0.3
            frontier_marker.color.r = 0.0
            frontier_marker.color.g = 1.0
            frontier_marker.color.b = 1.0  # Cyan - distinct from artifacts
            frontier_marker.color.a = 0.8
            
            for frontier in frontiers:
                point = Point()
                point.x, point.y, point.z = frontier[0], frontier[1], 0.2
                frontier_marker.points.append(point)
            
            marker_array.markers.append(frontier_marker)
        
        # Selected goal as yellow arrow (very distinct)
        if selected_frontier:
            goal_marker = Marker()
            goal_marker.header.frame_id = "map"
            goal_marker.header.stamp = self.get_clock().now().to_msg()
            goal_marker.ns = "exploration_goal"
            goal_marker.id = 0
            goal_marker.type = Marker.ARROW  # Different shape
            goal_marker.action = Marker.ADD
            
            goal_marker.pose.position.x = selected_frontier[0]
            goal_marker.pose.position.y = selected_frontier[1]
            goal_marker.pose.position.z = 1.0  # Higher than artifacts
            goal_marker.pose.orientation.w = 1.0
            
            goal_marker.scale.x = 2.0
            goal_marker.scale.y = 0.5
            goal_marker.scale.z = 0.5
            
            goal_marker.color.r = 1.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0  # Yellow - very distinct
            goal_marker.color.a = 1.0
            
            self.goal_marker_pub_.publish(goal_marker)
        
        self.frontier_marker_pub_.publish(marker_array)
    
    ################################
    # END OF PLANNING TASK 1: FRONTIER EXPLORATION METHODS
    ################################

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """
        
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Updated for autonomous frontier-based exploration!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        elif not self.exploration_complete_:
            ################################
            # PLANNING TASK 1: USE FRONTIER EXPLORATION FOR AUTONOMOUS MAPPING
            ################################
            self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
            ################################
        else:
            # Fallback to random goals when exploration is complete
            self.get_logger().info('Exploration complete! Switching to random goals.')
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        elif self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
            success = self.planner_frontier_exploration()
            if not success:
                # Fallback to random goal if frontier exploration fails
                self.get_logger().warn('Frontier exploration failed, falling back to random goal')
                self.planner_type_ = PlannerType.RANDOM_GOAL
                self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################
    # Removed diagnostic methods - not needed for simple geometric approach

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)
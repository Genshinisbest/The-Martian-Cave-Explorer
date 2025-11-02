#!/usr/bin/env python3

# Cave explorer  improved perception 3 + planner robustness (applied patches)
# - stable artifact confirmation & EMA smoothing
# - vision suppression around confirmed artifacts while inspecting
# - random goal sampling from occupancy grid free cells
# - supports all 6 artifact types, but planner inspects only stop_sign + green_crystal

import math
import random
from enum import Enum
import os
import datetime
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

def wrap_angle(angle):
    while angle < 0.0:
        angle += 2 * math.pi
    while angle > 2 * math.pi:
        angle -= 2 * math.pi
    return angle

def pose2d_to_pose(pose_2d):
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
    INSPECT_ARTIFACT = 7

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')
        # --- basic state ---
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]
        self.artifact_found_ = False

        # --- dataset / perception ---
        self.save_interval = 3.0
        self.last_saved_time = 0.0
        self.image_counter = 0
        self.custom_model = None
        self.use_custom_model = False

        try:
            if torch is not None:
                model_path = '/home/student/ros_ws/CV_Model/yolov5/best.pt'
                repo_path = '/home/student/ros_ws/CV_Model/yolov5'
                if os.path.exists(model_path) and os.path.exists(repo_path):
                    import sys
                    cwd = os.getcwd()
                    os.chdir(repo_path)
                    sys.path.insert(0, repo_path)
                    # load with torch.hub (as you had before)
                    self.custom_model = torch.hub.load('.', 'custom', path='best.pt', source='local', trust_repo=True)
                    self.use_custom_model = True
                    os.chdir(cwd)
                    self.get_logger().info('Loaded custom model via torch.hub')
            else:
                self.get_logger().warn('Torch not available')
        except Exception as e:
            self.get_logger().error(f'Model load error: {e}')
            self.use_custom_model = False

        # -------------------------------
        # PERCEPTION TASK 3 data structures
        # - self.detected_artifacts_positions stores provisional & confirmed detections:
        #   (x, y, z, type, confidence, hits)
        # - artifact_types_supported includes the six required classes
        # -------------------------------
        self.detected_artifacts_positions = []  # list of tuples (x,y,z,type,conf,hits)
        self.artifact_clustering_threshold = 0.6  # meters (tighter clustering)
        self.ema_alpha = 0.25
        self.confirm_hits_required = 6

        self.artifact_type_colors = {
            'stop_sign': (1.0, 0.0, 0.0),
            'green_crystal': (0.0, 1.0, 0.0),
            'green_alien': (0.0, 0.8, 0.2),
            'white_sphere': (1.0, 1.0, 1.0),
            'mushrooms': (0.8, 0.4, 0.0),
            'formation': (0.5, 0.5, 0.8)
        }
        self.artifact_labels_supported = ['green_crystal', 'green_alien', 'stop_sign', 'mushrooms', 'formation', 'white_sphere']

        # artifacts_to_inspect_ now stores dicts: {'point':Point,'type':str}
        self.artifacts_to_inspect_ = []
        self.inspected_artifacts_ = []  # list of Point
        self.artifact_locations_ = []   # for confirmed markers only

        # suppression: after confirming an artifact and queuing for inspection, suppress
        # new detections within suppress_radius until robot moves away
        self.suppress_vision_until_far = None  # (x,y) or None
        self.suppress_radius = 2.0  # meters

        # Motion-gate to prevent duplicate artifact logging when stationary
        self.last_pose_for_detection = None
        self.min_move_before_new_detection = 0.8  # meters

        # current twist (subscribe to odom)
        self.current_twist = Twist()

        # --- planning state ---
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.current_map_ = None
        self.map_info_ = None
        self.exploration_complete_ = False

        # inspection timing
        self.currently_inspecting_ = False
        self.inspect_timeout_s = 30.0
        self.inspect_start_time = None

        # Marker setup
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # --- other nodes / subscriptions ---
        self.cv_bridge_ = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True

        # parameters
        self.declare_parameter('print_feedback', False)
        self.declare_parameter('initial_mode', 'random')

        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        # subscribe to odom for twist to gate detections during rotations
        self.odom_sub_ = self.create_subscription(Twist, 'odom', self.odom_callback, 10)

        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

    # ---------------------------
    # Pose helper
    # ---------------------------
    def get_pose_2d(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except TransformException:
            return None
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y
        qw = t.transform.rotation.w
        qz = t.transform.rotation.z
        pose.theta = wrap_angle(2. * math.acos(qw) if qz >= 0 else -2. * math.acos(qw))
        return pose

    # ---------------------------
    # Map callback (frontier)
    # ---------------------------
    def map_callback(self, map_msg: OccupancyGrid):
        map_origin = [map_msg.info.origin.position.x, map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width
        self.xlim_ = [map_origin[0], map_origin[0] + map_width * map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1] + map_height * map_resolution]
        # store occupancy grid and info for valid-goal selection
        self.current_map_ = np.array(map_msg.data).reshape((map_height, map_width))
        self.map_info_ = map_msg.info

    # ---------------------------
    # Image callback & detection
    # ---------------------------
    def image_callback(self, image_msg):
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Save dataset images
        self.save_images_for_dataset(image)

        # Run detection (your trained model)
        detections = self.detect_artifacts_advanced(image)

        # Pre-filter: image dims
        img_h, img_w = image.shape[0], image.shape[1]

        # Convert detections to 3D positions (and cluster)
        artifact_positions = []
        for det in detections:
            x, y, w, h, conf, atype = det

            # Only allow known six types for Perception 3
            if atype not in self.artifact_labels_supported:
                continue

            # Reject bbox too close to image edges (junk)
            cx = x + w / 2.0
            if cx < 0.07 * img_w or cx > 0.93 * img_w:
                continue

            # Low confidence rejections
            type_thresholds = {
                'stop_sign': 0.4,
                'green_crystal': 0.15,
                'white_sphere': 0.25,
                'mushrooms': 0.15,
                'formation': 0.6,
                'green_alien': 0.15
            }
            if conf < type_thresholds.get(atype, 0.2):
                continue

            # If vision suppression active, and robot still near suppression anchor -> skip detection
            if self.suppress_vision_until_far is not None:
                pose2d = self.get_pose_2d()
                if pose2d:
                    ax, ay = self.suppress_vision_until_far
                    if math.hypot(pose2d.x - ax, pose2d.y - ay) < self.suppress_radius:
                        # skip adding any detection while near confirmed artifact
                        continue
                    else:
                        # moved away enough -> clear suppression
                        self.suppress_vision_until_far = None

            # Convert to world coordinates using geometric heuristic
            cx_int = int(cx)
            cy_int = int(y + h/2.0)
            pos3 = self.get_artifact_3d_position(cx_int, cy_int, h)
            if pos3:
                artifact_positions.append((pos3, atype, conf))

        # Update flag & add detections for clustering / markers
        self.artifact_found_ = len(artifact_positions) > 0
        for pos3, atype, conf in artifact_positions:
            self.add_detected_artifact(pos3, atype, conf)

        # Visualization of detections on image (still useful)
        for det in detections:
            x, y, w, h, conf, atype = det
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{atype}:{conf:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        msg = self.cv_bridge_.cv2_to_imgmsg(image, encoding='rgb8')
        self.image_detections_pub_.publish(msg)

        # Optional higher-level logging / localisation
        if self.artifact_found_:
            self.localise_artifact()

    # ---------------------------
    # Save dataset frames
    # ---------------------------
    def save_images_for_dataset(self, image):
        now = time.time()
        if now - self.last_saved_time >= self.save_interval:
            save_dir = '/home/student/ros_ws/dataset_images'
            os.makedirs(save_dir, exist_ok=True)
            fname = f"img_{self.image_counter:04d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            path = os.path.join(save_dir, fname)
            cv2.imwrite(path, image)
            self.image_counter += 1
            self.last_saved_time = now

    # ---------------------------
    # Detection using custom model
    # ---------------------------
    def detect_artifacts_advanced(self, image):
        detections = []
        if self.use_custom_model and self.custom_model:
            try:
                results = self.custom_model(image)
                model_dets = results.xyxy[0].cpu().numpy()
                labels = self.artifact_labels_supported
                for d in model_dets:
                    x1, y1, x2, y2, conf, cls = d
                    if conf >= 0.05:
                        cid = int(cls)
                        if cid < len(labels):
                            detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1), float(conf), labels[cid]))
            except Exception as e:
                self.get_logger().error(f'Detection error: {e}')
        return detections

    # ---------------------------
    # Get 3D pos estimate from pixel and box height
    # ---------------------------
    def get_artifact_3d_position(self, pixel_x, pixel_y, box_height):
        """Simple geometric approximation using robot pose + bounding box height -> distance"""
        robot_pose = self.get_pose_2d()
        if not robot_pose:
            return None

        image_width = 640
        fov_deg = 70.0
        offset_x = pixel_x - (image_width / 2)
        direction = robot_pose.theta + (offset_x / image_width) * math.radians(fov_deg)

        # heuristic distance estimate from box height (works often in simulation)
        if box_height and box_height > 0:
            est_dist = 220.0 / float(box_height)   # tunable constant from earlier code
            est_dist = max(0.8, min(12.0, est_dist))
        else:
            est_dist = 4.0

        ax = robot_pose.x + est_dist * math.cos(direction)
        ay = robot_pose.y + est_dist * math.sin(direction)
        return (ax, ay, 0.5)

    # ---------------------------
    # Add / cluster detections & track stable artifacts
    # ---------------------------
    def add_detected_artifact(self, pos3, atype, conf):
        """
        Add or update detected artifact in self.detected_artifacts_positions.
        We cluster by spatial proximity and use confidence-weighted averaging (EMA).
        Only when hits >= confirm_hits_required do we consider the detection confirmed.
        """

        # If robot is rotating in place, ignore vision (avoids spurious detections while pivoting)
        try:
            vx = self.current_twist.linear.x
            vth = self.current_twist.angular.z
        except Exception:
            vx = 0.0
            vth = 0.0

        if abs(vx) < 0.05 and abs(vth) > 0.3:
            # mostly rotating -> skip detection
            return

        # ---- MOTION-GATE: only accept detections when robot has moved a minimum distance ----
        pose = self.get_pose_2d()
        if pose and self.last_pose_for_detection:
            moved = math.hypot(
                pose.x - self.last_pose_for_detection.x,
                pose.y - self.last_pose_for_detection.y
            )
            if moved < self.min_move_before_new_detection:
                # haven't moved enough since last accepted detection -> skip
                return
        # update last motion checkpoint (store a copy)
        if pose:
            self.last_pose_for_detection = Pose2D(x=pose.x, y=pose.y, theta=pose.theta)

        if pos3 is None:
            return

        x, y, z = pos3

        # short names
        cluster_radius = self.artifact_clustering_threshold
        confirm_hits_required = self.confirm_hits_required
        ema_alpha = self.ema_alpha

        # Try to match with existing provisional/confirmed artifacts
        match_idx = None
        best_dist = float('inf')

        for i, art in enumerate(self.detected_artifacts_positions):
            px, py, pz, patype, pconf, phits = art
            if patype != atype:
                continue
            dist = math.hypot(px - x, py - y)
            if dist < cluster_radius and dist < best_dist:
                best_dist = dist
                match_idx = i

        if match_idx is not None:
            px, py, pz, patype, pconf, phits = self.detected_artifacts_positions[match_idx]
            # EMA smoothing towards new measurement
            nx = ema_alpha * x + (1.0 - ema_alpha) * px
            ny = ema_alpha * y + (1.0 - ema_alpha) * py
            nz = ema_alpha * z + (1.0 - ema_alpha) * pz
            nhits = phits + 1
            nconf = max(pconf, conf)
            self.detected_artifacts_positions[match_idx] = (nx, ny, nz, patype, nconf, nhits)

            # When newly confirmed, add to inspection queue (only for chosen types)
            if nhits == confirm_hits_required:
                # create Point and attach type
                p = Point(); p.x = nx; p.y = ny; p.z = nz
                # Check duplication in inspected or queued lists
                already_inspected = any(math.hypot(nx - q.x, ny - q.y) < 1.0 for q in self.inspected_artifacts_)
                already_queued = any(math.hypot(nx - q['point'].x, ny - q['point'].y) < 1.0 for q in self.artifacts_to_inspect_)
                # Add to confirmed marker list
                if not any(math.hypot(nx - q.x, ny - q.y) < 0.5 for q in self.artifact_locations_):
                    self.artifact_locations_.append(p)
                if not already_inspected and not already_queued:
                    # For planner we only queue the chosen inspection types (green_crystal, stop_sign)
                    if patype in ['green_crystal', 'stop_sign']:
                        self.artifacts_to_inspect_.append({'point': p, 'type': patype})
                        self.get_logger().warn(f"CONFIRMED {patype} at ({nx:.2f},{ny:.2f}) - queued for inspection")
                        # suppress vision near this confirmed artifact until we move away
                        self.suppress_vision_until_far = (nx, ny)
                    else:
                        self.get_logger().info(f"Confirmed (no-inspect) {patype} at ({nx:.2f},{ny:.2f})")
        else:
            # New provisional detection: store hits=1 (do NOT append to artifact_locations_ yet)
            self.detected_artifacts_positions.append((x, y, z, atype, conf, 1))

        # publish markers after every update (only confirmed will appear as stable markers)
        self.publish_artifact_markers()

    def odom_callback(self, msg):
        # we only need twist to detect rotation vs translation
        # if msg is Twist (we subscribed to Twist directly), use it; otherwise adapt accordingly
        if isinstance(msg, Twist):
            self.current_twist = msg
        else:
            # some systems may publish nav_msgs/Odometry instead; handle gracefully if that happens
            try:
                self.current_twist = msg.twist.twist
            except Exception:
                pass

    # ---------------------------
    # Localise artifact - cleaned printout for confirmed artifacts
    # ---------------------------
    def localise_artifact(self):
        if not self.detected_artifacts_positions:
            return
        confirmed = [(x,y,z,a,c,h) for (x,y,z,a,c,h) in self.detected_artifacts_positions if h >= max(3, self.confirm_hits_required//2)]
        if confirmed:
            self.get_logger().info('--- Confirmed Artifacts ---')
            for (x,y,z,a,c,h) in confirmed:
                self.get_logger().info(f'{a} @ ({x:.2f},{y:.2f}) hits={h}')
            self.get_logger().info('---------------------------')

    # ---------------------------
    # Publish RViz markers grouped by type (only confirmed show as spheres)
    # ---------------------------
    def publish_artifact_markers(self):
        marray = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Confirmed items (hits >= confirm_hits_required)
        confirmed = [(x,y,z,atype,conf,hits) for (x,y,z,atype,conf,hits) in self.detected_artifacts_positions if hits >= self.confirm_hits_required]

        # Create one sphere marker per confirmed artifact
        marker_id = 0
        for (x,y,z,atype,conf,hits) in confirmed:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'confirmed_artifacts'
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.9
            m.scale.y = 0.9
            m.scale.z = 0.9
            r,g,b = self.artifact_type_colors.get(atype,(1.0,1.0,1.0))
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
            marray.markers.append(m)
            marker_id += 1

        # Fallback: if nothing confirmed, we do NOT flood markers with every provisional detection
        # but still optionally show a minimal provisional group (we avoid doing this frequently)
        if not confirmed and self.artifact_locations_:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'provisional_artifacts'
            m.id = marker_id
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color.r = 0.2; m.color.g = 0.8; m.color.b = 0.2; m.color.a = 0.7
            m.points = self.artifact_locations_
            marray.markers.append(m)

        self.marker_pub_.publish(marray)

    # ---------------------------
    # Navigation helpers & callbacks
    # ---------------------------
    def planner_go_to_pose2d(self, pose2d):
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose = pose2d_to_pose(pose2d)
        self.goal_pose_vis_.publish(goal.pose)
        fb = self.feedback_callback if self.get_parameter('print_feedback').value else None
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        # mark that we're busy so main_loop won't spam new goals
        self.ready_for_next_goal_ = False
        try:
            self.send_goal_future_ = self.nav2_action_client_.send_goal_async(goal, feedback_callback=fb)
            self.send_goal_future_.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')
            self.ready_for_next_goal_ = True

    def goal_response_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn('Goal rejected by server')
            self.ready_for_next_goal_ = True
            return
        self.get_result_future_ = gh.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, fb_msg):
        fb = fb_msg.feedback
        # optionally show remaining distance
        if self.get_parameter('print_feedback').value:
            self.get_logger().info(f'{fb.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        # navigation finished (either success or failed)  allow next goal
        self.ready_for_next_goal_ = True

    # ---------------------------
    # Valid random goal sampling using occupancy grid (avoid walls)
    # ---------------------------
    def pick_random_valid_goal(self, min_distance=4.0, max_tries=80):
        """
        Picks a random free-world coordinate from current_map_ and map_info_.
        Enforces min_distance from robot pose and checks occupancy == 0 (free).
        Returns (x,y) or None.
        """
        if self.current_map_ is None or self.map_info_ is None:
            return None

        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return None

        height, width = self.current_map_.shape
        res = self.map_info_.resolution
        ox = self.map_info_.origin.position.x
        oy = self.map_info_.origin.position.y

        tries = 0
        while tries < max_tries:
            tries += 1
            # pick uniformly from known map bounding box
            gx = random.uniform(ox, ox + width * res)
            gy = random.uniform(oy, oy + height * res)

            # enforce min_distance
            if math.hypot(gx - robot_pose.x, gy - robot_pose.y) < min_distance:
                continue

            # map index
            cx = int((gx - ox) / res)
            cy = int((gy - oy) / res)
            if 0 <= cx < width and 0 <= cy < height:
                val = self.current_map_[cy, cx]
                if val == 0:  # free cell
                    return (gx, gy)
        return None

    def planner_random_goal(self):
        # Try to pick a valid free-space goal from occupancy grid
        pick = self.pick_random_valid_goal()
        if pick is None:
            # fallback: handpicked goals (as before)
            random_goals = [[15.2, 2.2],[30.7, 2.2],[43.0, 11.3],[10.0,10.0]]
            gx, gy = random.choice(random_goals)
        else:
            gx, gy = pick
        pose = Pose2D(x=gx, y=gy, theta=random.uniform(0, 2*math.pi))
        self.planner_go_to_pose2d(pose)

    # ---------------------------
    # Planner for inspection (unchanged logic but uses artifacts_to_inspect_ with type)
    # ---------------------------
    def planner_inspect_artifact(self):
        """
        Use the first queued artifact (self.artifacts_to_inspect_) as target,
        build a standoff pose 1.0 m back from the artifact, and send nav goal.
        On arrival, the main_loop will mark artifact as inspected and remove from queue.
        """
        if not self.artifacts_to_inspect_:
            # nothing to inspect
            self.ready_for_next_goal_ = True
            return

        target_dict = self.artifacts_to_inspect_[0]
        target = target_dict['point']
        target_type = target_dict['type']

        robot_pose = self.get_pose_2d()
        if not robot_pose:
            self.ready_for_next_goal_ = True
            return

        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        angle = math.atan2(dy, dx)
        standoff = 1.0  # meters
        ax = target.x - standoff * math.cos(angle)
        ay = target.y - standoff * math.sin(angle)

        self.currently_inspecting_ = True
        self.inspect_start_time = time.time()
        self.planner_go_to_pose2d(Pose2D(x=ax, y=ay, theta=angle))

    # ---------------------------
    # Main loop: orchestrates planners
    # ---------------------------
    def main_loop(self):
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform...')
            return

        # If a goal is currently running, check for inspection timeout and return
        if not self.ready_for_next_goal_:
            # if we are inspecting, apply timeout handling
            if self.currently_inspecting_ and self.inspect_start_time and time.time() - self.inspect_start_time > self.inspect_timeout_s:
                self.get_logger().warn('Inspection timeout - abandoning current target')
                self.currently_inspecting_ = False
                if self.artifacts_to_inspect_:
                    # remove the timed-out target and allow next goal
                    dropped = self.artifacts_to_inspect_.pop(0)
                    self.get_logger().info(f'Removed timed-out inspection target {dropped["type"]}')
                    self.ready_for_next_goal_ = True
            return

        # If we just finished an inspection goal, mark visited
        if self.currently_inspecting_ and self.ready_for_next_goal_:
            if self.artifacts_to_inspect_:
                v = self.artifacts_to_inspect_.pop(0)
                self.inspected_artifacts_.append(v['point'])
                self.get_logger().info(f'Inspection complete. Marked visited ({v["point"].x:.2f},{v["point"].y:.2f}) type={v["type"]}')
            self.currently_inspecting_ = False

        # ----- FINISH INSPECTION WHEN CLOSE ENOUGH -----
        if self.currently_inspecting_ and self.artifacts_to_inspect_:
            pose = self.get_pose_2d()
            target = self.artifacts_to_inspect_[0]['point']
            if pose and target:
                dist = math.hypot(target.x - pose.x, target.y - pose.y)
                if dist < 1.2:  # reached artifact zone
                    v = self.artifacts_to_inspect_.pop(0)
                    self.inspected_artifacts_.append(v['point'])
                    self.get_logger().info(
                        f" Reached artifact. Inspection complete at ({v['point'].x:.2f},{v['point'].y:.2f}) type={v['type']}"
                    )
                    self.currently_inspecting_ = False
                    self.ready_for_next_goal_ = True
                    return

        # ---------------------------
        # ROBUST PLANNING FSM
        # 1) If artifacts queued -> inspection (priority)
        # 2) Otherwise -> random exploration
        # ---------------------------

        # 1) INSPECTION HAS PRIORITY
        if self.artifacts_to_inspect_:
            # if not currently navigating, send inspection goal
            if not self.currently_inspecting_ and self.ready_for_next_goal_:
                self.planner_type_ = PlannerType.INSPECT_ARTIFACT
                self.currently_inspecting_ = True
                self.inspect_start_time = time.time()
                self.get_logger().info("*** Starting inspection mode ***")
                self.planner_inspect_artifact()
                return
            # Already going to artifact  just wait
            self.get_logger().info("INSPECTING  holding state")
            return

        # 2) NORMAL EXPLORATION MODE
        if self.ready_for_next_goal_:
            self.planner_type_ = PlannerType.RANDOM_GOAL
            self.get_logger().info("Exploring... sending random goal")
            self.planner_random_goal()
            return

        # 3) OTHERWISE: WAIT
        self.get_logger().info("WAITING for current navigation to finish")
        return


def main(args=None):
    rclpy.init(args=args)
    node = CaveExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

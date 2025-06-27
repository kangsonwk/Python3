#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: kangson896
@email:kangson896@gmail.com
@Version:0.1
@Date: 2025-06-27
@Description: 
"""

# Import common packages
from loguru import logger
import os
import sys
import cv2
import numpy as np
import pandas as pd
import pyautogui
import time
import math
import pyrealsense2 as rs
import mediapipe as mp
import subprocess
import re

# Import packages about PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap,QKeySequence
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QShortcut
)

# Import packages about OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Import custom packages or custom functions
# custom settings
os.environ.update(
    {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/home/p30016215206/anaconda3/envs/ER/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms"
    }
)
pyautogui.FAILSAFE = False
logger.add(f'./test.log',enqueue=True)

def line_plane_intersection(point1, point2):
    """
    Calculate the intersection point between a straight line formed by two three-dimensional points and the XY plane (z=0)
    : param point1: The first point (x1, y1, z1)
    : param point2: The Second point (x2, y2, z2)
    Return: Intersection coordinates (x, y, 0) or None (when the line is parallel to the plane)
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # Calculate direction vector
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    # Handling special situations: lines parallel to the plane
    if abs(dz) < 1e-10:  # Use small amounts to avoid floating point errors
        if abs(z1) < 1e-10:
            # Line in plane, return any point (here returns the first point)
            return (x1, y1, 0.0)
        else:
            # The line is parallel to the plane but not within the plane
            return None
    
    # Calculate Parameters t: z = z1 + t*dz = 0
    t = -z1 / dz
    
    # Calculate intersection coordinates
    x = x1 + t * dx
    y = y1 + t * dy
    
    return (x, y, 0.0)

def get_monitor_info(monitor_index=0):
    # Using xrandr command to obtain raw data
    output = subprocess.check_output(["xrandr"]).decode()
    
    # Analyze the connected monitor
    connected = [line.split()[0] for line in output.splitlines() 
                if " connected" in line]
    
    # Analyze the resolution of each monitor
    resolutions = {}
    for display in connected:
        res_match = re.search(r'(\d+x\d+)\+\d+\+\d+', output)
        if res_match:
            resolutions[display] = res_match.group(1)
    
    return connected, resolutions

def create_sphere_from_3_points(p1, p2, p3):
    # Vector computation
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)  # Plane normal vector
    
    # Solving linear equations
    A = np.array([
        [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1]), 2*(p2[2]-p1[2])],
        [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1]), 2*(p3[2]-p1[2])],
        [n[0], n[1], n[2]]
    ])
    b = np.array([
        p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2 + p2[2]**2 - p1[2]**2,
        p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2 + p3[2]**2 - p1[2]**2,
        n[0]*p1[0] + n[1]*p1[1] + n[2]*p1[2]
    ])
    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(p1 - center)
    return center, radius

class EyeGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.eye_points = []
        self.rotation = [0, 0, 0]
        self.pupil_offset = [0, 0]  # [x, y] offset for pupil movement

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, self.width() / self.height(), 0.1, 50.0)
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        if len(self.eye_points) >= 6:
            self.draw_eye_model()

    def draw_eye_model(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 5, 1])

        # Draw left eye
        glPushMatrix()
        glTranslatef(-0.3, 0, 0)

        # Eyeball (white part)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.9, 0.9, 0.95, 1.0])
        glutSolidSphere(0.2, 32, 32)

        # Iris (colored part)
        glPushMatrix()
        iris_max_offset = 1
        glTranslatef(
            iris_max_offset * self.pupil_offset[0], 
            iris_max_offset * self.pupil_offset[1], 
            0.18 - abs(iris_max_offset * self.pupil_offset[0]) * 0.5 - abs(iris_max_offset * self.pupil_offset[1]) * 0.5
        )
        
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.4, 0.2, 0.8, 1.0])
        glutSolidSphere(0.1, 32, 32)

        # Pupil (black center)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.1, 0.1, 0.1, 1.0])
        glutSolidSphere(0.05, 32, 32)
        glPopMatrix()

        glPopMatrix()

        # Draw right eye
        glPushMatrix()
        glTranslatef(0.3, 0, 0)

        # Eyeball (white part)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.9, 0.9, 0.95, 1.0])
        glutSolidSphere(0.2, 32, 32)

        # Iris (colored part)
        glPushMatrix()
        iris_max_offset =1
        glTranslatef(
            iris_max_offset * self.pupil_offset[0], 
            iris_max_offset * self.pupil_offset[1], 
            0.18 - abs(iris_max_offset * self.pupil_offset[0]) * 0.5 - abs(iris_max_offset * self.pupil_offset[1]) * 0.5
        )
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.4, 0.2, 0.8, 1.0])
        glutSolidSphere(0.1, 32, 32)

        # Pupil (black center)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.1, 0.1, 0.1, 1.0])
        glutSolidSphere(0.05, 32, 32)
        glPopMatrix()

        glPopMatrix()

        glDisable(GL_LIGHTING)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

class Eye3DWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Eye Model")
        self.setGeometry(1200, 100, 800, 600)

        self.gl_widget = EyeGLWidget()
        self.setCentralWidget(self.gl_widget)

        # Mouse interaction variables
        self.last_pos = None
        self.zoom_level = 5.0

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()

            if event.buttons() == Qt.LeftButton:
                self.gl_widget.rotation[0] += dy * 0.5
                self.gl_widget.rotation[1] += dx * 0.5
                self.gl_widget.updateGL()

        self.last_pos = event.pos()

    def wheelEvent(self, event):
        self.zoom_level -= event.angleDelta().y() * 0.001
        self.zoom_level = max(1.0, min(self.zoom_level, 10.0))
        self.gl_widget.zoom = self.zoom_level
        self.gl_widget.updateGL()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_camera()
        self.setup_mediapipe()
        self.current_eye_points = []
        self.eye_3d_window = Eye3DWindow()
        self.eye_3d_window.show()
        
        # Mouse control parameters
        self.start_time = None
        self.last_eye_pos = None
        self.tolerance = 0.05  # Tolerance for changes in eye position
        self.hold_duration = 3 # Duration of trigger click (seconds)
        self.last_screen_pos = None  # The former screen coordinates
        self.smoothed_pos = None  # Smooth coordinates
        
        # Eye-to-screen mapping parameters
        self.scale_x = 400  # Horizontal scaling factor
        self.scale_y = 400  # Vertical scaling factor
        self.deadzone = 0.02  # Minimum movement threshold
        
        # Screen parameters
        self.screen_w, self.screen_h = pyautogui.size()
        self.screen_w_mm,self.screen_h_mm=600,340
        self.PPI_x = round(self.screen_w_mm/self.screen_w,2)
        self.PPI_y = round(self.screen_h_mm/self.screen_h,2)
        self.depth_camera_screen_z_distance = 30
        
        # Set shortcut
        self.init_shortcut()
        
    def init_shortcut(self):
        self.restart_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.restart_shortcut.activated.connect(
            self.close_event)
            
    def close_event(self):
        self.eye_3d_window.close()
        self.close()
        
    def setup_ui(self):
        self.setWindowTitle("D435i Eye Tracking")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.camera_view = QLabel()
        self.depth_view = QLabel()

        # Distance display
        self.distance_label = QLabel("Pupil Distance: Not detected")
        self.distance_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        layout.addWidget(self.camera_view)
        layout.addWidget(self.depth_view)
        layout.addWidget(self.distance_label)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)

    def setup_mediapipe(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Initial head pose parameter
        self.head_pose = [0, 0, 0]  # pitch, yaw, roll

    def estimate_head_pose(self, face_landmarks, image_shape):
        """Estimate head pose"""
        image_height, image_width = image_shape[:2]
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # The tip of the nose
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left corner of left eye
            (225.0, 170.0, -135.0),      # Right corner of right eye
            (-150.0, -150.0, -125.0),    # Left corner of the mouth
            (150.0, -150.0, -125.0)      # Right corner of the mouth
        ])
        
        # 2D image points
        image_points = np.array([
            (face_landmarks.landmark[4].x * image_width, face_landmarks.landmark[4].y * image_height),     # 鼻尖
            (face_landmarks.landmark[152].x * image_width, face_landmarks.landmark[152].y * image_height), # 下巴
            (face_landmarks.landmark[133].x * image_width, face_landmarks.landmark[133].y * image_height), # 左眼左角
            (face_landmarks.landmark[362].x * image_width, face_landmarks.landmark[362].y * image_height), # 右眼右角
            (face_landmarks.landmark[61].x * image_width, face_landmarks.landmark[61].y * image_height),   # 左嘴角
            (face_landmarks.landmark[291].x * image_width, face_landmarks.landmark[291].y * image_height)  # 右嘴角
        ], dtype="double")
        
        # Camera parameters
        focal_length = image_width
        center = (image_width/2, image_height/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1)) # Assuming there is no lens distortion
        
        # Solving attitude
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            # Convert the rotation vector into a rotation matrix
            rmat, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles from the rotation matrix
            sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(rmat[2,1], rmat[2,2])
                y = math.atan2(-rmat[2,0], sy)
                z = math.atan2(rmat[1,0], rmat[0,0])
            else:
                x = math.atan2(-rmat[1,2], rmat[1,1])
                y = math.atan2(-rmat[2,0], sy)
                z = 0
                
            # Convert to angle
            self.head_pose = [
                math.degrees(x),  # pitch
                math.degrees(y),  # yaw 
                math.degrees(z)   # roll
            ]
            self.eye_3d_window.gl_widget.rotation = self.head_pose

    def update_frame(self):
        """Update camera frames and process eye tracking"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return

            # Process eye data and get screen position
            self.update_eye_data(depth_frame, color_frame)

            # Display RGB image
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.camera_view.setPixmap(QPixmap.fromImage(q_img))

        except Exception as e:
            logger.error(f"Error in update_frame: {e}")
            return

    def smooth_mouse_movement(self, new_pos):
        # Smooth/Movement parameters
        SMOOTHING_FACTOR = 0.5  # Smoothing coefficient (0-1)
        MIN_MOVE_THRESHOLD = 5  # Minimum movement threshold (pixels)
        # Ensure that the input coordinates are valid firstly
        if not new_pos or len(new_pos) != 2:
            return None
        try:
            new_x, new_y = int(new_pos[0]), int(new_pos[1])
            if self.smoothed_pos is None:
                self.smoothed_pos = (new_x, new_y)
            else:
                # Using exponential smoothing algorithm
                smoothed_x = int(self.smoothed_pos[0] * (1 - SMOOTHING_FACTOR) + new_x * SMOOTHING_FACTOR)
                smoothed_y = int(self.smoothed_pos[1] * (1 - SMOOTHING_FACTOR) + new_y * SMOOTHING_FACTOR)
                
                # Ensure that the coordinates are within the screen range
                smoothed_x = max(0, min(self.screen_w-1, smoothed_x))
                smoothed_y = max(0, min(self.screen_h-1, smoothed_y))
                
                self.smoothed_pos = (smoothed_x, smoothed_y)
            return self.smoothed_pos
        except Exception as e:
            logger.error(f"Error in smooth_mouse_movement: {e}")
            return None

    def update_eye_data(self, depth_frame, color_frame):
        """Update eye tracking data and synchronize with 3D window"""
        try:
            if color_frame is None:
                return

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            self.current_eye_points = []
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                height, width = color_image.shape[:2]
        except Exception as e:
            logger.error(f"Error processing eye data: {e}")
            return
            
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            height, width = color_image.shape[:2]
            
            # Estimate head pose
            self.estimate_head_pose(face_landmarks, color_image.shape)
            
            # Obtain the key points of the left and right pupils
            left_pupil_key_points = []
            right_pupil_key_points = []
            left_iris = right_iris = None
            
            for i in range(468, 478):
                landmark = face_landmarks.landmark[i]
                if i == 468:  # Right eye iris center
                    right_iris = (landmark.x, landmark.y)
                elif 468 < i < 473 and len(right_pupil_key_points) < 3:
                    right_pupil_key_points.append((landmark.x, landmark.y))    
                elif i == 473:  # Left eye iris center
                    left_iris = (landmark.x, landmark.y)
                elif i > 473 and len(left_pupil_key_points) < 3:
                    left_pupil_key_points.append((landmark.x, landmark.y))
            
            # Draw red pupil keypoints (left and right pupil centers) on the image
            if left_iris:
                px = int(left_iris[0] * width)
                py = int(left_iris[1] * height)
                cv2.circle(rgb_image, (px, py), 3, (0, 0, 255), -1)
            
            if right_iris:
                px = int(right_iris[0] * width)
                py = int(right_iris[1] * height)
                cv2.circle(rgb_image, (px, py), 3, (0, 0, 255), -1)
            
            # Calculate 3D position and line of sight intersection point
            if left_iris and right_iris and len(left_pupil_key_points) == 3 and len(right_pupil_key_points) == 3:
                    # Obtain the 3D coordinates of the left eye
                    left_points = []
                    for i, (x, y) in enumerate(left_pupil_key_points):
                        px = int(x * width)
                        py = int(y * height)
                        depth = depth_frame.get_distance(px, py)
                        if depth > 0:
                            point = rs.rs2_deproject_pixel_to_point(
                                depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                                [px, py],
                                depth
                            )
                            left_points.append(np.array(point))
                    
                    # Obtain the 3D coordinates of the right eye
                    right_points = []
                    for i, (x, y) in enumerate(right_pupil_key_points):
                        px = int(x * width)
                        py = int(y * height)
                        depth = depth_frame.get_distance(px, py)
                        if depth > 0:
                            point = rs.rs2_deproject_pixel_to_point(
                                depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                                [px, py],
                                depth
                            )
                            right_points.append(np.array(point))
                    
                    # Obtain the 3D coordinates of the pupil center
                    left_iris_pixel_x = int(left_iris[0] * width)
                    left_iris_pixel_y = int(left_iris[1] * height)
                    left_iris_depth = depth_frame.get_distance(left_iris_pixel_x, left_iris_pixel_y)
                    right_iris_pixel_x = int(right_iris[0] * width)
                    right_iris_pixel_y = int(right_iris[1] * height)
                    right_iris_depth = depth_frame.get_distance(right_iris_pixel_x, right_iris_pixel_y)
                    
                    if (len(left_points) >= 3 and len(right_points) >= 3 and left_iris_depth > 0 and right_iris_depth > 0):
                        
                        left_iris_location = rs.rs2_deproject_pixel_to_point(
                            depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                            [left_iris_pixel_x, left_iris_pixel_y],
                            left_iris_depth
                        )
                        
                        right_iris_location = rs.rs2_deproject_pixel_to_point(
                            depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                            [right_iris_pixel_x, right_iris_pixel_y],
                            right_iris_depth
                        )
                        
                        # Build a left eye sphere model and obtain the center of the sphere
                        left_center, left_radius = create_sphere_from_3_points(
                            left_points[0], left_points[1], left_points[2])
                        
                        # Build a rightt eye sphere model and obtain the center of the sphere
                        right_center, right_radius = create_sphere_from_3_points(
                            right_points[0], right_points[1], right_points[2])
                        
                        # Calculate the direction of the line of sight (from the center of the sphere to the center of the pupil)
                        left_gaze = np.array(left_iris_location) - left_center
                        right_gaze = np.array(right_iris_location) - right_center
                        
                        # Calculate the intersection point between the line of sight and the screen (using the left eye line of sight this time)
                        screen_intersection = line_plane_intersection(
                            left_center,
                            left_center + left_gaze
                        )
                        
                        logger.info(f'left_iris_location:{left_iris_location}')
                        logger.info(f'right_iris_location:{right_iris_location}')
                        logger.info(f'screen_intersection:{screen_intersection}')
                        if screen_intersection:
                            # Adjust screen coordinates based on head posture
                            head_pitch_factor = 1 + (self.head_pose[0] / 30)  # pitch影响垂直方向
                            head_yaw_factor = 1 + (self.head_pose[1] / 30)    # yaw影响水平方向
                            
                            try:
                                # Map 3D intersection points to screen coordinates, considering head posture
                                screen_x = int((screen_intersection[0] * self.screen_w / self.screen_w_mm * head_yaw_factor) + self.screen_w/2)
                                screen_y = int((-screen_intersection[1] * self.screen_h / self.screen_h_mm * head_pitch_factor) + self.screen_h/2)
                                
                                # Limit coordinates within the screen range
                                screen_x = max(0, min(self.screen_w-1, screen_x))
                                screen_y = max(0, min(self.screen_h-1, screen_y))
                                
                                # Ensure that the coordinates are valid
                                if math.isnan(screen_x) or math.isnan(screen_y):
                                    raise ValueError("Invalid screen coordinates")
                            except Exception as e:
                                logger.error(f"Error calculating screen coordinates: {e}")
                                return
                            # The function of else in try-except-else
                            # (1) Execute only when there are no exceptions in the try block
                            # (2) Suitable for placing code that depends on the successful execution of try blocks
                            # (3) Separating from try block code can improve readability
                            else:
                                # Move the mouse smoothly
                                smoothed_pos = self.smooth_mouse_movement((screen_x, screen_y))
                                if smoothed_pos:
                                    pyautogui.moveTo(smoothed_pos[0], smoothed_pos[1])
                            
                            # Update the 3D eye model
                            self.eye_3d_window.gl_widget.pupil_offset = [
                                (left_iris[0] - 0.5) * 2,
                                (left_iris[1] - 0.5) * 2
                            ]
                            self.eye_3d_window.gl_widget.updateGL()
                            
                            # Update distance display
                            distance = math.sqrt(
                                (left_iris_location[0] - right_iris_location[0])**2 +
                                (left_iris_location[1] - right_iris_location[1])**2 +
                                (left_iris_location[2] - right_iris_location[2])**2
                            )
                            self.distance_label.setText(f"Pupil Distance: {distance:.2f}m")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.timer.start(30)  # Update at ~30fps
    sys.exit(app.exec_())

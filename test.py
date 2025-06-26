import sys
import cv2
import numpy as np
import os
import pyrealsense2 as rs
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

os.environ.update(
    {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/home/p30016215206/anaconda3/envs/ER/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms"
    }
)


class EyeTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_camera()
        self.setup_mediapipe()

    def setup_ui(self):
        self.setWindowTitle("D435i 眼部距离检测")
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QWidget()
        layout = QVBoxLayout()

        # 双视图显示区域
        self.raw_view = QLabel()
        self.result_view = QLabel()
        layout.addWidget(self.raw_view)
        layout.addWidget(self.result_view)

        # 控制按钮
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 按钮事件
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(config)

    def setup_mediapipe(self):
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.7)

    def calculate_distance(self, depth_frame, point):
        return depth_frame.get_distance(point[0], point[1])

    def update_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 人脸检测
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_image)

        annotated_image = color_image.copy()

        if results.detections:
            for detection in results.detections:
                # 获取眼部关键点
                keypoints = detection.location_data.relative_keypoints
                left_eye = self.mp_face.FaceKeyPoint.LEFT_EYE
                right_eye = self.mp_face.FaceKeyPoint.RIGHT_EYE

                # 转换坐标
                def get_pixel_coord(kp):
                    return int(kp.x * color_image.shape[1]), int(
                        kp.y * color_image.shape[0]
                    )

                left_eye_pixel = get_pixel_coord(keypoints[left_eye])
                right_eye_pixel = get_pixel_coord(keypoints[right_eye])

                # 计算距离
                left_dist = self.calculate_distance(depth_frame, left_eye_pixel)
                right_dist = self.calculate_distance(depth_frame, right_eye_pixel)
                avg_dist = (left_dist + right_dist) / 2

                # 绘制标记
                cv2.circle(annotated_image, left_eye_pixel, 5, (0, 255, 0), -1)
                cv2.circle(annotated_image, right_eye_pixel, 5, (0, 255, 0), -1)
                cv2.putText(
                    annotated_image,
                    f"Distance: {avg_dist:.2f}m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        # 显示图像
        self.display_image(color_image, self.raw_view)
        self.display_image(annotated_image, self.result_view)

    def display_image(self, img, label):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))

    def start_detection(self):
        self.timer.start(30)

    def stop_detection(self):
        self.timer.stop()
        self.pipeline.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTracker()
    window.show()
    sys.exit(app.exec_())

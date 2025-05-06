import cv2
import mediapipe as mp
import pyautogui

# 初始化MediaPipe面部关键点检测
# test git
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1
)

# 获取屏幕尺寸
screen_w, screen_h = pyautogui.size()

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 定义关键点索引（MediaPipe面部网格模型中的左右眼关键点）
LEFT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145]  # 左眼周围关键点
RIGHT_EYE_POINTS = [362, 385, 386, 263, 373, 374, 380]  # 右眼周围关键点


def map_eye_to_screen(eye_center, frame_w, frame_h):
    """
    将眼部中心坐标映射到屏幕坐标（简单线性映射）
    """
    x = int((eye_center[0] / frame_w) * screen_w)
    y = int((eye_center[1] / frame_h) * screen_h)
    return x, y


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 镜像翻转图像
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测面部关键点
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # 提取左右眼关键点
        left_eye = [
            (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
            for i in LEFT_EYE_POINTS
        ]
        right_eye = [
            (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
            for i in RIGHT_EYE_POINTS
        ]

        # 计算眼部中心坐标（取关键点平均值）
        left_center = (
            sum([p[0] for p in left_eye]) // len(left_eye),
            sum([p[1] for p in left_eye]) // len(left_eye),
        )
        right_center = (
            sum([p[0] for p in right_eye]) // len(right_eye),
            sum([p[1] for p in right_eye]) // len(right_eye),
        )

        # 映射到屏幕坐标
        gaze_x, gaze_y = map_eye_to_screen(left_center, frame_w, frame_h)

        # 在屏幕上显示聚焦点（使用pyautogui绘制）
        pyautogui.moveTo(gaze_x, gaze_y)

        # 在摄像头画面中绘制眼部关键点（可选）
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # 显示摄像头画面
    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

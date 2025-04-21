import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# ================== 配置区域 ==================
MODEL_PATH = "666.h5"  # 训练好的多分类模型路径
LABELS = ["drink", "read", "play_with_computer"]  # 必须与训练时的标签顺序一致
SEQUENCE_LENGTH = 30  # 必须与训练时一致
NUM_KEYPOINTS = 13  # 与训练时一致的上半身关键点数量
NUM_FEATURES = 4  # 每个关键点的特征数 (x, y, z, visibility)
MIN_DETECTION_CONFIDENCE = 0.7  # 姿态检测置信度阈值
MIN_PREDICTION_CONFIDENCE = 0.7  # 动作判定阈值

# MediaPipe Pose的13个上半身关键点索引（与训练时一致）
UPPER_BODY_KEYPOINTS = [
    0,  # nose
    1, 2, 3, 4,  # left_eye, right_eye, left_ear, right_ear
    11, 12,  # left_shoulder, right_shoulder
    13, 14,  # left_elbow, right_elbow
    15, 16,  # left_wrist, right_wrist
    23, 24  # left_hip, right_hip
]


# =============================================

class MultiActionDetector:
    def __init__(self):
        # 加载模型
        self.model = load_model(MODEL_PATH)
        print(f"模型加载完成! 可识别的动作: {LABELS}")

        # 初始化MediaPipe Pose
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )

        # 初始化序列缓冲区
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.current_action = None
        self.action_counter = 0
        self.action_durations = {label: 0.0 for label in LABELS}

        # 初始化标签计数器
        self.action_counts = {label: 0 for label in LABELS}  # 计数器初始化

    def process_frame(self, frame):
        """处理单帧并返回带标注的画面"""
        # 1. 姿态估计
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            self._draw_status(frame, "NO PERSON DETECTED", (0, 0, 255))
            return frame

        # 2. 提取并处理关键点（与训练时一致）
        kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark])

        # 选择上半身关键点
        upper_kpts = kpts[UPPER_BODY_KEYPOINTS]

        # 标准化处理（与训练时一致）
        left_hip = kpts[23]
        right_hip = kpts[24]
        torso_center = (left_hip[:3] + right_hip[:3]) / 2
        upper_kpts[:, :3] -= torso_center[:3]

        # 展平特征（保持与训练相同的处理顺序）
        kpts_flat = upper_kpts.flatten()
        self.sequence.append(kpts_flat)

        # 3. 绘制姿态关键点
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # 4. 当序列足够时进行预测
        if len(self.sequence) == SEQUENCE_LENGTH:
            # 转换为模型输入格式（与训练时一致）
            input_data = np.array(self.sequence).reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_FEATURES)

            predictions = self.model.predict(input_data, verbose=0)[0]

            # 5. 获取最高置信度的动作
            max_idx = np.argmax(predictions)
            max_prob = predictions[max_idx]

            if max_prob > MIN_PREDICTION_CONFIDENCE:
                detected_action = LABELS[max_idx]

                # 动作持续计数（避免闪烁）
                if detected_action == self.current_action:
                    self.action_counter += 1
                else:
                    self.action_counter = 0
                    self.current_action = detected_action

                # 仅当连续检测到5次相同动作时才更新显示
                if self.action_counter >= 5:
                    self._draw_status(frame, f"{detected_action.upper()}: {max_prob:.2f}", (0, 255, 0))

                    # 增加动作计数器
                    self.action_counts[detected_action] += 1

            else:
                self._draw_status(frame, "UNKNOWN ACTION", (255, 165, 0))
        else:
            self._draw_status(frame, f"Collecting Frames: {len(self.sequence)}/{SEQUENCE_LENGTH}", (255, 255, 0))

        # 显示各个动作的计数
        self._draw_action_counts(frame)

        return frame

    def _draw_status(self, frame, text, color):
        """在画面顶部绘制状态信息"""
        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _draw_action_counts(self, frame):
        """在画面上绘制每个动作的计数"""
        y_offset = 70  # 从下方50像素开始显示
        for action, count in self.action_counts.items():
            cv2.putText(frame, f"{action}: {int(count/10)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            y_offset += 30  # 每个动作间隔30像素

def main():
    detector = MultiActionDetector()
    cap = cv2.VideoCapture(0)  # 使用默认摄像头

    # 设置摄像头分辨率（建议与训练数据采集时一致）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 校验摄像头是否成功打开
    if not cap.isOpened():
        print("Error: 无法打开摄像头。")
        return

    print("开始视频流，按ESC退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: 读取视频帧失败。")
            break

        # 镜像画面（更自然的用户体验）
        frame = cv2.flip(frame, 1)

        # 检测动作
        processed_frame = detector.process_frame(frame)

        # 显示结果
        cv2.imshow('Multi-Action Recognition', processed_frame)

        # 按ESC退出
        if cv2.waitKey(1) == 27:
            print("退出程序...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

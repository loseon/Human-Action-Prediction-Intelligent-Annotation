import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import json
import os
import logging
import traceback
from tqdm import tqdm

# ================== 配置区域 ==================
VIDEO_PATH = r"E:\ai+robot\move_data_video"  # 视频根目录
OUTPUT_ROOT = "annotations"  # 标注根目录
LABELS = ["drink", "read","play_with_computer"]  # 动作标签列表
SEQUENCE_LENGTH = 30  # 分析帧窗口大小
LOG_LEVEL = logging.INFO  # 日志级别
# =============================================

# 配置日志系统
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('annotation.log'),
        logging.StreamHandler()
    ]
)

class VideoAnnotator:
    def __init__(self):
        # 初始化MediaPipe模型
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.7
            )
            self.last_valid_kpts = None
            self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
            self.global_counter = 0  # 新增全局计数器
        except Exception as e:
            logging.error(f"MediaPipe初始化失败: {str(e)}")
            raise

    def _get_enhanced_keypoints(self, frame):
        """增强版关键点检测（带异常值过滤）"""
        try:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                logging.debug("未检测到人体姿态")
                return None

            kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in results.pose_landmarks.landmark])

            # 肢体长度校验
            left_arm = np.linalg.norm(kpts[11, :2] - kpts[13, :2])
            right_arm = np.linalg.norm(kpts[12, :2] - kpts[14, :2])
            length_diff = abs(left_arm - right_arm) / max(left_arm, right_arm)

            if length_diff > 0.4:
                logging.warning(f"肢体长度异常差异: {length_diff:.2f}, 使用上一帧数据")
                return self.last_valid_kpts

            self.last_valid_kpts = kpts
            return kpts

        except Exception as e:
            logging.error(f"关键点提取失败: {str(e)}")
            return None

    def _save_data(self, frame, kpts, label, video_path):
        """保存帧图像和标注文件（使用全局计数器）"""
        try:
            # 创建分类目录
            label_dir = os.path.join(OUTPUT_ROOT, label)
            frame_dir = os.path.join(label_dir, "frame")
            json_dir = os.path.join(label_dir, "json")

            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)

            # 使用全局计数器生成唯一ID
            unique_id = self.global_counter
            self.global_counter += 1  # 计数器递增

            # 保存图像帧
            frame_path = os.path.join(frame_dir, f"{unique_id:08d}.jpg")
            cv2.imwrite(frame_path, frame)

            # 保存标注JSON
            json_path = os.path.join(json_dir, f"{unique_id:08d}.json")
            with open(json_path, 'w') as f:
                json.dump({
                    'unique_id': unique_id,
                    'original_frame_id': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    'label': label,
                    'keypoints': kpts.tolist(),
                    'source_video': os.path.basename(video_path)}
                    , f, indent=2)

            logging.debug(f"已保存: {frame_path} 和 {json_path}")
        except Exception as e:
            logging.error(f"文件保存失败: {str(e)}")

    def _process_single_video(self, video_path, label):
        """处理单个视频文件"""
        global cap  # 用于在_save_data中获取帧号
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name = os.path.basename(video_path)
            logging.info(f"开始处理视频: {video_name} (分类: {label}, 总帧数: {total_frames})")

            with tqdm(total=total_frames, desc=f"处理 {video_name}", unit="帧") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    try:
                        kpts = self._get_enhanced_keypoints(frame)
                        if kpts is None:
                            pbar.update(1)
                            continue

                        # 保存当前帧数据
                        self._save_data(frame, kpts, label, video_path)
                        pbar.update(1)

                    except Exception as e:
                        logging.error(f"{video_name} 第 {cap.get(cv2.CAP_PROP_POS_FRAMES)} 帧处理失败: {str(e)}")
                        pbar.update(1)
                        continue

            logging.info(f"视频 {video_name} 处理完成！共处理 {self.global_counter} 帧")

        except Exception as e:
            logging.error(f"处理视频 {video_path} 失败: {str(e)}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    def process_all_videos(self):
        """处理所有分类视频"""
        try:
            if not os.path.exists(VIDEO_PATH):
                raise FileNotFoundError(f"视频目录不存在: {VIDEO_PATH}")

            # 确保输出目录存在
            os.makedirs(OUTPUT_ROOT, exist_ok=True)

            # 处理每个分类目录
            for label in LABELS:
                label_path = os.path.join(VIDEO_PATH, label)
                if not os.path.exists(label_path):
                    logging.warning(f"分类目录不存在: {label_path}")
                    continue

                # 获取分类目录下所有视频文件
                video_files = [f for f in os.listdir(label_path)
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

                if not video_files:
                    logging.warning(f"分类目录 {label} 中没有视频文件")
                    continue

                logging.info(f"开始处理分类: {label} (共 {len(video_files)} 个视频)")

                # 处理该分类下的所有视频
                for video_file in video_files:
                    video_path = os.path.join(label_path, video_file)
                    self._process_single_video(video_path, label)

            logging.info(f"所有视频处理完成！总计处理 {self.global_counter} 帧")

        except Exception as e:
            logging.critical(f"处理流程异常: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        # Windows系统DPI适配
        if os.name == 'nt':
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)

        annotator = VideoAnnotator()
        annotator.process_all_videos()
    except Exception as e:
        logging.critical(f"程序崩溃: {str(e)}")
        traceback.print_exc()
    finally:
        input("按回车键退出...")
import os
import cv2
import json
import numpy as np
from utils import load_json
from algos import CombinedDetector, TableSeg
from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer

# 初始化流式推理框架
dispatcher = DevelopDispatcher.create(mode="offline", buffer=1)
inferencer = Inference(dispatcher)

class EnhancedRunner:
    def __init__(self, dispatcher: DevelopDispatcher, inferencer: Inference, config_path: str):
        self.dispatcher = dispatcher
        self.inferencer = inferencer
        self.config = self._validate_config(config_path)

    def _validate_config(self, config_path: str) -> dict:
        """验证并加载配置文件"""
        cfg = load_json(config_path)
        required_keys = ["yolo_model", "mmpose_config", "mmpose_weights"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise KeyError(f"配置缺少必要参数: {missing}")
        return cfg

    def _init_model(self):
        """加载双模型算法"""
        self.inferencer.load_algo(
            CombinedDetector(),
            frame_count=1,
            frame_step=0,
            interval=0.04,
            yolo_model_path=self.config["yolo_model"],
            mmpose_config=self.config["mmpose_config"],
            mmpose_weights=self.config["mmpose_weights"]
        )

    def process_video(self, input_path: str, output_path: str):
        """处理视频流"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入视频不存在: {input_path}")

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 初始化视频播放器
        player = Player(
            self.dispatcher,
            OpenCVProducer(width, height),  # 根据实际分辨率调整
            source=input_path
        )

        # 启动推理流水线，添加录制参数
        self.inferencer.start(
            player=player,
            fps=total_frames,  # 根据实际帧率调整
            position=0,
            mode="offline",
            recording_path=output_path  # 关键参数：保存视频
        )

    @inferencer.process
    def analyze_results(inference: Inference,  *args, **kwargs):
        """实时结果分析"""
        print("进入渲染")
        def _process_detections(metadata: dict):
            """处理检测元数据"""
            # YOLO检测结果解析
            for detection in metadata["yolo"]:
                print(f"[YOLO] 检测到 {detection['class_name']} "
                    f"置信度: {detection['confidence']:.2f} "
                    f"位置: {detection['bbox']}")

            # 姿态估计结果解析
            for idx, pose in enumerate(metadata["pose"]):
                print(f"[MMPose] 人员{idx} 关键点数量: {len(pose['keypoints'])} "
                    f"检测置信度: {pose['bbox_score']:.2f}")
        try:
            combined_data = inferencer.dispatcher.get_result("CombinedDetector", clear=False)
            print("combined_data:",len(combined_data))
            if combined_data:
                # for frame_key in combined_data:
                #     print("frame_key", frame_key)
                #     frame_data = combined_data[frame_key]
                frame_data = combined_data['0']
                if "metadata" in frame_data:
                    metadata = frame_data["metadata"]
                    if "yolo" in metadata and "pose" in metadata:
                        _process_detections(metadata)
                        return frame_data["processed_frame"]
                    else:
                        print("处理结果时出错: metadata 缺少 'yolo' 或 'pose' 字段")
                else:
                    print("处理结果时出错: frame_data 缺少 'metadata' 字段")
        except Exception as e:
            print(f"处理结果时出错: {e}")
            return None

    def _process_detections(self, metadata: dict):
        """处理检测元数据"""
        # YOLO检测结果解析
        for detection in metadata["yolo"]:
            print(f"[YOLO] 检测到 {detection['class_name']} "
                  f"置信度: {detection['confidence']:.2f} "
                  f"位置: {detection['bbox']}")

        # 姿态估计结果解析
        for idx, pose in enumerate(metadata["pose"]):
            print(f"[MMPose] 人员{idx} 关键点数量: {len(pose['keypoints'])} "
                  f"检测置信度: {pose['bbox_score']:.2f}")

if __name__ == "__main__":
    # 配置路径
    CONFIG_PATH = "model.json"
    INPUT_VIDEO = "/home/yangf/algo_yolo/algo_all/video_data/pose_whiteman.mp4"
    OUTPUT_VIDEO = "results/processed_pose_whiteman2-1.mp4"

    # 初始化运行器
    runner = EnhancedRunner(
        dispatcher=dispatcher,
        inferencer=inferencer,
        config_path=CONFIG_PATH
    )

    # 加载模型
    runner._init_model()

    # 启动处理
    try:
        runner.process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    except KeyboardInterrupt:
        print("用户中断处理")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
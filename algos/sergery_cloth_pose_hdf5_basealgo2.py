import os

import numpy as np
from ultralytics import YOLO

from mmpose.apis import MMPoseInferencer
from stream_infer.algo import BaseAlgo
from stream_infer.log import logger


class CombinedDetector(BaseAlgo):
    def __init__(self):
        super().__init__(name="CombinedDetector")
        self.yolo_model = None
        self.mmpose_inferencer = None
        self._init_config()
        self.frame_counter = 0

    def _init_config(self):
        """初始化默认配置"""
        self.class_names = [
            "Mask",
            "Surgical gown",
            "Surgical cap",
            "Gloves",
            "Sterile gloves",
            "Operating gown",
            "Non-sterile hands",
            "Slippers",
            "No mask",
            "No surgical cap",
            "Scrub suit",
            "Shoe covers",
        ]
        self.palette = [
            (255, 56, 56),
            (56, 255, 56),
            (56, 56, 255),
            (255, 156, 56),
            (156, 255, 56),
            (56, 156, 255),
            (255, 56, 156),
            (156, 56, 255),
            (255, 156, 156),
            (156, 255, 156),
            (156, 156, 255),
            (255, 255, 56),
        ]

    def init(
        self,
        yolo_model_path: str,
        mmpose_config: str,
        mmpose_weights: str,
    ):
        """初始化双模型

        Args:
            yolo_model_path: YOLO模型路径
            mmpose_config: MMPose配置文件路径
            mmpose_weights: MMPose权重文件路径
        """
        self._init_yolo(yolo_model_path)
        self._init_mmpose(mmpose_config, mmpose_weights)

    def _init_yolo(self, model_path: str):
        """加载YOLO模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO模型路径不存在: {model_path}")
        self.yolo_model = YOLO(model_path, verbose=False)
        logger.info(f"成功加载YOLO模型: {os.path.basename(model_path)}")

    def _init_mmpose(self, config_path: str, weights_path: str):
        """加载MMPose模型"""
        if not os.path.exists(weights_path):
            self._download_mmpose_weights(weights_path)

        self.mmpose_inferencer = MMPoseInferencer(
            pose2d=config_path, pose2d_weights=weights_path, device="cuda:0"
        )
        logger.info(f"成功加载MMPose模型: {os.path.basename(weights_path)}")

    def _download_mmpose_weights(self, weights_path: str):
        """自动下载权重文件"""
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        logger.warning(f"开始下载MMPose权重文件到: {weights_path}")
        os.system(
            f"wget -P {os.path.dirname(weights_path)} https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
        )
        if not os.path.exists(weights_path):
            raise RuntimeError("MMPose权重文件下载失败")

    def run(self, frames: list) -> dict:
        """执行双模型推理"""
        if len(frames) != 1:
            raise ValueError("当前版本只支持单帧处理")

        frame = frames[0]
        self.frame_counter += 1

        # YOLO目标检测
        yolo_results = self.yolo_model(frame, verbose=False)

        # MMPose关键点检测
        mmpose_results = self._process_mmpose(frame)

        return {
            "frame_id": self.frame_counter,
            "yolo": self._parse_yolo_results(yolo_results),
            "pose": mmpose_results,
        }

    def _parse_yolo_results(self, results):
        """解析YOLO检测结果"""
        output = []
        for r in results:
            for box in r.boxes:
                item = {
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                }
                output.append(item)
        return output

    def _process_mmpose(self, frame: np.ndarray) -> list:
        """处理姿态估计结果"""
        result_generator = self.mmpose_inferencer(
            frame, det_bbox_thr=0.6, return_vis=False
        )

        try:
            result = next(result_generator)
            return result["predictions"][0] if "predictions" in result else []
        except StopIteration:
            return []


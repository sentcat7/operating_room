import os
import cv2
import numpy as np
import h5py
from tqdm import tqdm
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

        # 关键点可视化配置
        self.COCO_KEYPOINT_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        self.KEYPOINT_COLORS = {
            "head": (240, 80, 40),
            "left": (0, 165, 255),
            "right": (0, 255, 0)
        }
        self.KEYPOINT_PART_MAPPING = {
            0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
            5: "left", 6: "right", 7: "left", 8: "right", 9: "left",
            10: "right", 11: "left", 12: "right", 13: "left", 14: "right",
            15: "left", 16: "right"
        }

    def _init_config(self):
        """初始化默认配置"""
        self.class_names = [
            "Mask", "Surgical gown", "Surgical cap", "Gloves",
            "Sterile gloves", "Operating gown", "Non-sterile hands",
            "Slippers", "No mask", "No surgical cap", "Scrub suit", "Shoe covers"
        ]
        self.palette = [
            (255, 56, 56), (56, 255, 56), (56, 56, 255),
            (255, 156, 56), (156, 255, 56), (56, 156, 255),
            (255, 56, 156), (156, 56, 255), (255, 156, 156),
            (156, 255, 156), (156, 156, 255), (255, 255, 56)
        ]

    def init(self, yolo_model_path: str, mmpose_config: str, mmpose_weights: str):
        """初始化双模型"""
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
            pose2d=config_path,
            pose2d_weights=weights_path,
            device='cuda:0'
        )
        logger.info(f"成功加载MMPose模型: {os.path.basename(weights_path)}")

    def _download_mmpose_weights(self, weights_path: str):
        """自动下载权重文件"""
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        logger.warning(f"开始下载MMPose权重文件到: {weights_path}")
        os.system(f"wget -P {os.path.dirname(weights_path)} https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth")
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
        processed_frame = self._draw_yolo_results(frame.copy(), yolo_results)
        
        # MMPose关键点检测
        mmpose_results = self._process_mmpose(frame)
        processed_frame = self._draw_keypoints(processed_frame, mmpose_results)
        
        return {
            "processed_frame": processed_frame,
            "metadata": {
                "frame_id": self.frame_counter,
                "yolo": self._parse_yolo_results(yolo_results),
                "pose": mmpose_results
            }
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
                    "class_name": self.class_names[int(box.cls[0])]
                }
                output.append(item)
        return output

    def _process_mmpose(self, frame: np.ndarray) -> list:
        """处理姿态估计结果"""
        result_generator = self.mmpose_inferencer(
            frame,
            det_bbox_thr=0.6,
            return_vis=False
        )
        
        try:
            result = next(result_generator)
            return result['predictions'][0] if 'predictions' in result else []
        except StopIteration:
            return []

    def _draw_yolo_results(self, frame: np.ndarray, results) -> np.ndarray:
        """绘制YOLO检测框"""
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()

                color = self.palette[cls_id % len(self.palette)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{self.class_names[cls_id]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def _draw_keypoints(self, frame: np.ndarray, results: list) -> np.ndarray:
        """绘制关键点"""
        for person in results:
            if person.get('bbox_score', 0) >= 0.6:
                keypoints = person.get('keypoints', [])
                self._draw_skeleton(frame, keypoints)
        return frame

    def _draw_skeleton(self, frame: np.ndarray, keypoints: list):
        """绘制骨骼连线"""
        for pair in self.COCO_KEYPOINT_CONNECTIONS:
            idx1, idx2 = pair
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                x1, y1 = map(int, keypoints[idx1])
                x2, y2 = map(int, keypoints[idx2])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for idx, (x, y) in enumerate(keypoints):
            if idx in self.KEYPOINT_PART_MAPPING:
                part = self.KEYPOINT_PART_MAPPING[idx]
                color = self.KEYPOINT_COLORS[part]
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                cv2.putText(frame, str(idx), (int(x)+10, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def test_video(self, input_path: str, output_path: str = None, save_hdf5: bool = False):
        """视频测试入口"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 初始化输出
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            logger.info(f"初始化视频写入器: {output_path}")

        # 准备数据存储
        all_metadata = []
        progress = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 执行推理
                result = self.run([frame])
                processed_frame = result["processed_frame"]

                # 保存结果
                if writer:
                    writer.write(processed_frame)
                if save_hdf5:
                    all_metadata.append(result["metadata"])

                progress.update(1)

        finally:
            # 释放资源
            if cap: cap.release()
            if writer: writer.release()
            progress.close()

            # 保存数据
            if save_hdf5 and all_metadata:
                self._save_to_hdf5(all_metadata, os.path.dirname(output_path))
                logger.info(f"分析数据已保存到HDF5文件")

    def _save_to_hdf5(self, metadata_list: list, output_dir: str):
        """保存分析结果到HDF5"""
        os.makedirs(output_dir, exist_ok=True)
        hdf5_path = os.path.join(output_dir, "analysis_results.h5")

        with h5py.File(hdf5_path, 'w') as hdf:
            # 创建元数据组
            meta_group = hdf.create_group("metadata")
            meta_group.attrs["total_frames"] = len(metadata_list)

            # 逐帧保存数据
            for idx, data in enumerate(metadata_list):
                frame_group = hdf.create_group(f"frame_{idx:06d}")
                self._create_yolo_dataset(frame_group, data["yolo"])
                self._create_pose_dataset(frame_group, data["pose"])

    def _create_yolo_dataset(self, parent_group, yolo_data):
        """保存YOLO检测结果"""
        yolo_group = parent_group.create_group("yolo_detections")
        for i, detection in enumerate(yolo_data):
            det_group = yolo_group.create_group(f"det_{i:04d}")
            det_group.create_dataset("bbox", data=detection["bbox"])
            det_group.attrs["class_id"] = detection["class_id"]
            det_group.attrs["class_name"] = detection["class_name"]
            det_group.attrs["confidence"] = detection["confidence"]

    def _create_pose_dataset(self, parent_group, pose_data):
        """保存姿态估计结果"""
        pose_group = parent_group.create_group("pose_estimations")
        for i, person in enumerate(pose_data):
            person_group = pose_group.create_group(f"person_{i:04d}")
            if "keypoints" in person:
                person_group.create_dataset("keypoints", data=person["keypoints"])
            if "keypoint_scores" in person:
                person_group.create_dataset("scores", data=person["keypoint_scores"])
            person_group.attrs["bbox_score"] = person.get("bbox_score", 0)

if __name__ == "__main__":
    # 配置参数
    # 模型路径配置（请根据实际情况修改）
    YOLO_MODEL = "/home/yangf/yy_algo/yolo8/ultralytics/runs/detect/yolo8_cloth_detect_netsurgey_source2/weights/best.pt"
    MMPOSE_CONFIG = "/home/yangf/mm_lab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    MMPOSE_WEIGHTS = "/home/yangf/mm_lab/mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

    # 输入输出配置
    INPUT_VIDEO = "/home/yangf/algo_yolo/algo_all/video_data/pose_cloth_all.mp4"
    OUTPUT_VIDEO = "results/processed_pose_cloth_all.mp4"

    # 初始化检测器
    detector = CombinedDetector()
    detector.init(YOLO_MODEL, MMPOSE_CONFIG, MMPOSE_WEIGHTS)

    # 执行视频处理
    detector.test_video(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        save_hdf5=True
    )
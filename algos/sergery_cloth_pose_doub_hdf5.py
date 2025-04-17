import os
import cv2
import numpy as np
import h5py
from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
import numpy as np


# 自定义 JSON 编码器，用于处理 numpy 类型
class NumpyEncoder(np.ndarray):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class CombinedDetector:
    def __init__(self):
        self.yolo_model = None
        self.mmpose_inferencer = None
        self.class_names = ["Mask", "Surgical gown", "Surgical cap",
                            "Gloves", "Sterile gloves", "Operating gown",
                            "Non - sterile hands", "Slippers", "No mask",
                            "No surgical cap", "Scrub suit", "Shoe covers"]

        # 关键点配置
        self.COCO_KEYPOINT_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
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

    def init(self, yolo_model_path, mmpose_config, mmpose_weights):
        self._init_yolo(yolo_model_path)
        self._init_mmpose(mmpose_config, mmpose_weights)

    def _init_yolo(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO模型未找到: {model_path}")
        self.yolo_model = YOLO(model_path)

    def _init_mmpose(self, config_path, weights_path):
        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            os.system(f"wget -P {os.path.dirname(weights_path)} https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth")
        self.mmpose_inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=weights_path,
            device='cuda:0'
        )

    def _draw_yolo_results(self, frame, results):
        palette = [
            (255, 56, 56), (56, 255, 56), (56, 56, 255),
            (255, 156, 56), (156, 255, 56), (56, 156, 255),
            (255, 56, 156), (156, 56, 255), (255, 156, 156),
            (156, 255, 156), (156, 156, 255), (255, 255, 56)
        ]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()

                color = palette[cls_id % len(palette)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{self.class_names[cls_id]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def _draw_keypoints(self, frame, keypoints, scores):
        for pair in self.COCO_KEYPOINT_CONNECTIONS:
            idx1, idx2 = pair
            x1, y1 = map(int, keypoints[idx1])
            x2, y2 = map(int, keypoints[idx2])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for idx, (x, y) in enumerate(keypoints):
            part = self.KEYPOINT_PART_MAPPING[idx]
            color = self.KEYPOINT_COLORS[part]
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
            cv2.putText(frame, str(idx), (int(x) + 10, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run(self, frame):
        if frame is None:
            raise ValueError("输入帧为空")

        # 目标检测
        yolo_results = self.yolo_model(frame, verbose=False)
        frame = self._draw_yolo_results(frame, yolo_results)

        yolo_boxes = []
        yolo_classes = []
        yolo_confs = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                yolo_boxes.append([x1, y1, x2, y2])
                yolo_classes.append(cls_id)
                yolo_confs.append(conf)

        # 关键点检测
        result_generator = self.mmpose_inferencer(
            frame,
            show=False,
            out_dir=None,
            det_bbox_thr=0.6,
            draw_bbox=False,
            draw_keypoint=False,
            return_vis=False
        )
        try:
            result = next(result_generator)
        except StopIteration:
            return frame, {'yolo_boxes': np.array(yolo_boxes), 'yolo_classes': np.array(yolo_classes),
                           'yolo_confs': np.array(yolo_confs), 'predictions': []}

        keypoints_list = []
        scores_list = []
        # 绘制关键点
        if 'predictions' in result:
            for person in result['predictions'][0]:
                bbox_score = person.get('bbox_score', 0)
                if bbox_score >= 0.6:  # 检查检测框置信度
                    keypoints = person.get('keypoints', [])
                    scores = person.get('keypoint_scores', [])
                    if len(keypoints) > 0 and len(scores) > 0:
                        frame = self._draw_keypoints(frame, keypoints, scores)
                        keypoints_list.append(keypoints)
                        scores_list.append(scores)

        result['yolo_boxes'] = np.array(yolo_boxes)
        result['yolo_classes'] = np.array(yolo_classes)
        result['yolo_confs'] = np.array(yolo_confs)
        result['keypoints'] = np.array(keypoints_list)
        result['keypoint_scores'] = np.array(scores_list)

        return frame, result

    def save_results_to_hdf5(self, all_results, output_dir):
        hdf5_file = h5py.File(os.path.join(output_dir, "all_frames_prediction_results.h5"), 'w')
        for frame_count, result in enumerate(all_results):
            frame_group = hdf5_file.create_group(f"frame_{frame_count}")

            # 保存 YOLO 结果
            yolo_group = frame_group.create_group("yolo_results")
            yolo_group.create_dataset("boxes", data=result['yolo_boxes'])
            yolo_group.create_dataset("classes", data=result['yolo_classes'])
            yolo_group.create_dataset("confs", data=result['yolo_confs'])

            # 保存关键点结果
            keypoints_group = frame_group.create_group("keypoints_results")
            keypoints_group.create_dataset("keypoints", data=result['keypoints'])
            keypoints_group.create_dataset("keypoint_scores", data=result['keypoint_scores'])

            # 保存 MMPose 原始预测结果
            if isinstance(result['predictions'], list):
                for person_index, person_prediction in enumerate(result['predictions']):
                    if isinstance(person_prediction, dict):
                        person_group = keypoints_group.create_group(f"person_{person_index}")
                        for key, value in person_prediction.items():
                            if isinstance(value, np.ndarray):
                                person_group.create_dataset(key, data=value)
                            else:
                                try:
                                    person_group.attrs[key] = value
                                except TypeError:
                                    print(f"无法将 {key} 的值保存到 HDF5 文件，类型为 {type(value)}")
                    else:
                        print(f"第 {frame_count} 帧的第 {person_index} 个人的预测结果不是字典类型，跳过保存。")
            else:
                print(f"第 {frame_count} 帧的预测结果不是列表类型，跳过保存。")

        hdf5_file.close()
        print(f"所有帧预测结果已保存至: {os.path.join(output_dir, 'all_frames_prediction_results.h5')}")

    def test_image(self, input_path, output_path="results/combined_image.jpg", save_results_flag=False):
        frame = cv2.imread(input_path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {input_path}")

        processed_frame, result = self.run(frame)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 确保输出文件扩展名为.jpg
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        output_path = os.path.join(output_dir, f"{output_name}.jpg")

        cv2.imwrite(output_path, processed_frame)
        print(f"自定义标注结果已保存至: {output_path}")

        if save_results_flag and result is not None:
            self.save_results_to_hdf5([result], output_dir)

    def test_video(self, input_path, output_path=None, save_results_flag=False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, result = self.run(frame)

            if output_path:
                out.write(processed_frame)
            else:
                cv2.imshow("Processed Frame", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_results_flag and result is not None:
                all_results.append(result)

            frame_count += 1

        cap.release()
        if output_path:
            out.release()
        else:
            cv2.destroyAllWindows()

        if save_results_flag and all_results:
            self.save_results_to_hdf5(all_results, output_dir)


if __name__ == "__main__":
    # 模型路径配置（请根据实际情况修改）
    YOLO_MODEL_PATH = "/home/yangf/yy_algo/yolo8/ultralytics/runs/detect/yolo8_cloth_detect_netsurgey_source2/weights/best.pt"
    MMPOSE_CONFIG = "/home/yangf/mm_lab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    MMPOSE_WEIGHTS = "/home/yangf/mm_lab/mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

    # 输入输出配置
    INPUT_SOURCE = "/home/yangf/algo_yolo/algo_all/video_data/pose_cloth_all.mp4"
    OUTPUT_PATH = "results/processed_pose_cloth_all.mp4"
    SAVE_RESULTS_FLAG = True

    # 初始化并运行
    detector = CombinedDetector()
    detector.init(YOLO_MODEL_PATH, MMPOSE_CONFIG, MMPOSE_WEIGHTS)

    # 处理图片或视频
    if INPUT_SOURCE.lower().endswith(('.jpg', '.jpeg', '.png')):
        detector.test_image(INPUT_SOURCE, OUTPUT_PATH, SAVE_RESULTS_FLAG)
    else:
        detector.test_video(INPUT_SOURCE, OUTPUT_PATH, SAVE_RESULTS_FLAG)

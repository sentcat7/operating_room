import os
import cv2
from utils import load_json
from algos import CombinedDetector
from stream_infer import Inference, Player
from stream_infer.dispatcher import DevelopDispatcher
from stream_infer.producer import OpenCVProducer
from stream_infer.log import logger

# 配置文件路径
CONFIG_PATH = "model.json"
INPUT_VIDEO = "/home/yangf/algo_yolo/algo_all/video_data/pose_whiteman.mp4"
OUTPUT_VIDEO = "results/processed_video.mp4"


# 加载配置
def validate_config(config_path: str) -> dict:
    cfg = load_json(config_path)
    required_keys = ["yolo_model", "mmpose_config", "mmpose_weights"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"配置缺少必要参数: {missing}")
    return cfg


# 初始化
dispatcher = DevelopDispatcher.create(mode="offline", buffer=1)
inference = Inference(dispatcher)

# 加载配置
config = validate_config(CONFIG_PATH)

# 加载算法
inference.load_algo(
    CombinedDetector(),
    frame_count=1,
    frame_step=0,
    interval=0.01,
    yolo_model_path=config["yolo_model"],
    mmpose_config=config["mmpose_config"],
    mmpose_weights=config["mmpose_weights"],
)


# 定义处理函数
@inference.process
def analyze_results(inference: Inference, *args, **kwargs):
    """实时结果分析"""
    # 获取当前帧
    frame = kwargs.get("frame")
    if frame is None:
        return None

    # 关键点可视化配置
    KEYPOINT_COLORS = {
        "head": (240, 80, 40),  # 头部关键点颜色（蓝红色）
        "left": (0, 165, 255),  # 左侧关键点颜色（橙色）
        "right": (0, 255, 0),  # 右侧关键点颜色（绿色）
    }

    KEYPOINT_PART_MAPPING = {
        0: "head",
        1: "head",
        2: "head",
        3: "head",
        4: "head",
        5: "left",
        6: "right",
        7: "left",
        8: "right",
        9: "left",
        10: "right",
        11: "left",
        12: "right",
        13: "left",
        14: "right",
        15: "left",
        16: "right",
    }

    COCO_KEYPOINT_CONNECTIONS = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    # 手术服装检测颜色配置
    PALETTE = [
        (255, 56, 56),  # 红色
        (56, 255, 56),  # 绿色
        (56, 56, 255),  # 蓝色
        (255, 156, 56),  # 橙色
        (156, 255, 56),  # 黄绿色
        (56, 156, 255),  # 浅蓝色
        (255, 56, 156),  # 粉色
        (156, 56, 255),  # 紫色
        (255, 156, 156),  # 浅红色
        (156, 255, 156),  # 浅绿色
        (156, 156, 255),  # 浅蓝色
        (255, 255, 56),  # 黄色
    ]

    def _draw_detections(frame, data):
        # 绘制YOLO检测框
        for detection in data["yolo"]:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, detection["bbox"])
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # 使用与原始代码相同的颜色映射
            color = PALETTE[class_id % len(PALETTE)]

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            label = f"{class_name} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )

        # 绘制姿态关键点
        for person in data["pose"]:
            if person.get("bbox_score", 0) >= 0.6 and "keypoints" in person:
                keypoints = person["keypoints"]

                # 绘制骨骼连线
                for pair in COCO_KEYPOINT_CONNECTIONS:
                    idx1, idx2 = pair
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        x1, y1 = map(int, keypoints[idx1])
                        x2, y2 = map(int, keypoints[idx2])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制关键点（使用与原始代码相同的颜色映射）
                for idx, (x, y) in enumerate(keypoints):
                    if idx in KEYPOINT_PART_MAPPING:
                        part = KEYPOINT_PART_MAPPING[idx]
                        color = KEYPOINT_COLORS[part]
                        cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                        cv2.putText(
                            frame,
                            str(idx),
                            (int(x) + 10, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
      # 绘制日志信息
        hand_waist_log = ""
        all_hands_above_head_log = ""
        for person in data["pose"]:
            if person.get("bbox_score", 0) >= 0.6 and "keypoints" in person:
                keypoints = person["keypoints"]
                if len(keypoints) >= 13:
                    left_hand_y = keypoints[9][1]
                    right_hand_y = keypoints[10][1]
                    left_waist_y = keypoints[11][1]
                    right_waist_y = keypoints[12][1]
                    head1_y = keypoints[1][1]
                    head2_y = keypoints[2][1]

                    # 检测某个手低于某个腰
                    if (left_hand_y > left_waist_y) or (left_hand_y > right_waist_y) or \
                       (right_hand_y > left_waist_y) or (right_hand_y > right_waist_y):
                        hand_waist_log = "hand lowed waist"

                    # 检测所有手高于1或2号点
                    if (left_hand_y < head1_y or left_hand_y < head2_y) and \
                       (right_hand_y < head1_y or right_hand_y < head2_y):
                        all_hands_above_head_log = "hand higher than head"

        if hand_waist_log:
            cv2.putText(frame, hand_waist_log, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if all_hands_above_head_log:
            cv2.putText(frame, all_hands_above_head_log, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _log_detections(data: dict):
        for detection in data["yolo"]:
            logger.debug(
                f"[YOLO] 检测到 {detection['class_name']} "
                f"置信度: {detection['confidence']:.2f} "
                f"位置: {detection['bbox']}"
            )

        for idx, pose in enumerate(data["pose"]):
            logger.debug(
                f"[MMPose] 人员{idx} 关键点数量: {len(pose.get('keypoints', []))} "
                f"检测置信度: {pose.get('bbox_score', 0):.2f}"
            )

    combined_data = inference.dispatcher.get_last_result("CombinedDetector", clear=True)
    if combined_data is not None:
        _, data = combined_data
        _log_detections(data)
        _draw_detections(frame, data)

    return frame

def _check_hand_waist_position(data):
    for person in data["pose"]:
        if person.get("bbox_score", 0) >= 0.6 and "keypoints" in person:
            keypoints = person["keypoints"]
            # 假设9和10是手的关键点，11和12是腰部的关键点
            if len(keypoints) >= 13:
                left_hand_y = keypoints[9][1]
                right_hand_y = keypoints[10][1]
                left_waist_y = keypoints[11][1]
                right_waist_y = keypoints[12][1]
                head1_y = keypoints[1][1]
                head2_y = keypoints[2][1]

                # 检测某个手低于某个腰
                if (left_hand_y > left_waist_y) or (left_hand_y > right_waist_y) or \
                (right_hand_y > left_waist_y) or (right_hand_y > right_waist_y):
                    logger.debug("某个手低于某个腰")

                # 检测所有手高于1或2号点
                if (left_hand_y < head1_y or left_hand_y < head2_y) and \
                (right_hand_y < head1_y or right_hand_y < head2_y):
                    logger.debug("所有手高于1或2号点")



if __name__ == "__main__":
    # 检查输入视频是否存在
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"输入视频不存在: {INPUT_VIDEO}")

    # 获取视频信息
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取实际帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # 创建播放器
    player = Player(
        dispatcher,
        OpenCVProducer(width, height),
        source=INPUT_VIDEO,
    )

    # 启动推理
    try:
        inference.start(
            player=player,
            fps=fps,  # 使用视频的实际帧率
            position=0,
            mode="offline",
            recording_path=OUTPUT_VIDEO,
            # logging_level="DEBUG",
        )

    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
    finally:
        logger.info(f"处理结果已保存至: {OUTPUT_VIDEO}")


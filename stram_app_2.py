import cv2
import streamlit as st
from mmengine.config import Config
from algos import CombinedDetector  # 从 offline_main5.py 中导入 CombinedDetector
from stream_infer import Inference, StreamlitApp
from stream_infer.dispatcher import DevelopDispatcher
from offline_main5 import validate_config, CONFIG_PATH  # 导入必要的函数和配置路径

# 全局变量用于记录已经加载的模型
loaded_models = {}


# 加载配置
config = validate_config(CONFIG_PATH)

# 加载mmpose配置文件
# mmpose_cfg = Config.fromfile(config["mmpose_config"])

dispatcher = DevelopDispatcher.create(mode="offline", buffer=5)
inference = Inference(dispatcher)


def load_model(model_path):
    global loaded_models
    if model_path in loaded_models:
        return loaded_models[model_path]
    # 这里假设 CombinedDetector 内部会根据传入的路径加载模型
    model = CombinedDetector()
    loaded_models[model_path] = model
    return model


# 加载 CombinedDetector 算法
model = load_model(config["yolo_model"] + config["mmpose_weights"])
inference.load_algo(
    model,
    frame_count=1,
    frame_step=0,
    interval=0.01,
    yolo_model_path=config["yolo_model"],
    mmpose_config=config["mmpose_config"],  # 使用加载后的Config对象
    mmpose_weights=config["mmpose_weights"],
)


app = StreamlitApp(inference)


# Set frame annotation func
@app.annotate_frame
def annotate_frame(app: StreamlitApp, name, data, frame):
    if name == "CombinedDetector":
        # 绘制YOLO检测框
        for detection in data["yolo"]:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, detection["bbox"])
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            # 使用与原始代码相同的颜色映射
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

        for person in data["pose"]:
            # print(data["pose"])
            if person.get("bbox_score", 0) >= 0.6 and "keypoints" in person:
                keypoints = person["keypoints"]
                keypoint_sc = person["keypoint_scores"]
                print("keypoint_scores",keypoint_sc)
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
    return frame


algo_containers = {}
algo_states = {}


# Set output display func
@app.output
def output(app: StreamlitApp, name, position, data):
    if data is None:
        return

    global algo_containers, algo_states
    if name not in algo_containers:
        algo_containers[name] = app.output_widgets[name].empty()
        algo_states[name] = {"last_had_data": False}

    algo_containers[name].empty()

    container = algo_containers[name].container()

    if name == "CombinedDetector":
        # 统计YOLO检测结果
        things = [detection["class_name"] for detection in data["yolo"]]

        if not things:
            container.subheader("未检测到物体")
            from pandas import DataFrame
            df = DataFrame({"物体类型": [], "数量": []})
            algo_states[name]["last_had_data"] = False
        else:
            thing_counts = {}
            for thing in things:
                if thing in thing_counts:
                    thing_counts[thing] += 1
                else:
                    thing_counts[thing] = 1

            container.subheader(f"检测到 {len(things)} 个物体")
            from pandas import DataFrame
            df = DataFrame(
                {
                    "物体类型": list(thing_counts.keys()),
                    "数量": list(thing_counts.values()),
                }
            )
            algo_states[name]["last_had_data"] = True

        container.dataframe(
            df,
            column_config={
                "物体类型": st.column_config.TextColumn("物体类型"),
                "数量": st.column_config.NumberColumn("数量", format="%d"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # 统计姿态检测结果
        keypoints = [person["keypoints"] for person in data["pose"] if person.get("bbox_score", 0) >= 0.6 and "keypoints" in person]
        # print(keypoints)
        num_persons = len(keypoints)

        summary = f"时间: {position}秒 | 检测到 {num_persons} 人"
        container.subheader(summary)

        if num_persons == 0:
            algo_states[name]["last_had_data"] = False
            return

        if num_persons > 0 and num_persons <= 3:
            for i, person in enumerate(keypoints[:3]):
                with container.expander(f"人物 #{i + 1} 详情", expanded=False):
                    valid_kps = 0
                    for kp in person:
                        if len(kp) >= 3:
                            confidence = kp[2]
                            st.write(f"人物 #{i + 1} 关键点 {kp} 置信度: {confidence}")
                            if confidence > 0.5:
                                valid_kps += 1
                    st.text(f"有效关键点: {valid_kps}/{len(person)}")
        algo_states[name]["last_had_data"] = num_persons > 0


app.start(producer_type="pyav", clear=False)  # options: opencv, pyav
    
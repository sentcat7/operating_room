import cv2
import streamlit as st
from mmengine.config import Config
from algos import CombinedDetector
from stream_infer import Inference, StreamlitApp
from stream_infer.dispatcher import DevelopDispatcher
import time


# 假设这个函数是从其他地方导入的，这里只是示例
def validate_config(CONFIG_PATH):
    config = {
        "tsm_weights_path": "checkpoint2/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar"
    }
    return config


CONFIG_PATH = "config.yaml"
# 加载配置
config = validate_config(CONFIG_PATH)

dispatcher = DevelopDispatcher.create(mode="offline", buffer=5)
inference = Inference(dispatcher)


def load_model(model_path):
    from stream_infer.algo import BaseAlgo
    if model_path not in BaseAlgo._instances:
        model = CombinedDetector()
        model.init(config["tsm_weights_path"])
        BaseAlgo._instances[model_path] = model
    return BaseAlgo._instances[model_path]


# 加载CombinedDetector算法
model = load_model(config["tsm_weights_path"])
inference.load_algo(
    model,
    frame_count=1,
    frame_step=0,
    interval=0.01,
    tsm_weights_path=config["tsm_weights_path"]
)

app = StreamlitApp(inference)


# Set frame annotation func
@app.annotate_frame
def annotate_frame(app: StreamlitApp, name, data, frame):
    if name == "CombinedDetector":
        action = data["action"]
        cv2.putText(frame, f"动作: {action}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        action = data["action"]
        container.subheader(f"检测到动作: {action}")
        algo_states[name]["last_had_data"] = True


app.start(producer_type="pyav", clear=False)
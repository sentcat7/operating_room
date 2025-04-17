import os
import time
import numpy as np
from stream_infer.algo import BaseAlgo
from stream_infer.log import logger
from ops.models import TSN
from ops.transforms import *
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
import torchvision


class CombinedDetector2(BaseAlgo):
    def __init__(self):
        super().__init__(name="CombinedDetector")
        self.tsm_model = None
        self.transform_hyj = None
        self.frame_counter = 0
        self._init_config()

    def _init_config(self):
        self.arch = 'resnet50'
        self.num_class = 3
        self.num_segments = 8
        self.modality = 'RGB'
        self.base_model = 'resnet50'
        self.consensus_type = 'avg'
        self.dataset = 'ucf101'
        self.dropout = 0.1
        self.img_feature_dim = 256
        self.no_partialbn = True
        self.pretrain = 'imagenet'
        self.shift = True
        self.shift_div = 8
        self.shift_place = 'blockres'
        self.temporal_pool = False
        self.non_local = False
        self.tune_from = None

        self.INFO_BAR_HEIGHT = 80
        self.FONT_SCALE_ACTION = 40
        self.FONT_SCALE_FPS = 25
        self.FONT_THICKNESS = 2
        self.TEXT_Y_OFFSET = 50
        self.TEXT_LEFT_MARGIN = 30
        self.TEXT_RIGHT_MARGIN = 30
        self.BACKGROUND_COLOR = (255, 255, 255)
        self.TEXT_COLOR = (0, 0, 0)

        self.FONT_PATH = 'simhei.ttf'
        self.font_action = ImageFont.truetype(self.FONT_PATH, self.FONT_SCALE_ACTION)
        self.font_fps = ImageFont.truetype(self.FONT_PATH, self.FONT_SCALE_FPS)

        self.cls_text = ["一号结", "二号结", "    "]

    def init(self, tsm_weights_path):
        self._init_tsm_model()
        self._load_tsm_weights(tsm_weights_path)

    def _init_tsm_model(self):
        self.tsm_model = TSN(self.num_class, self.num_segments, self.modality,
                             base_model=self.arch,
                             consensus_type=self.consensus_type,
                             dropout=self.dropout,
                             img_feature_dim=self.img_feature_dim,
                             partial_bn=not self.no_partialbn,
                             pretrain=self.pretrain,
                             is_shift=self.shift, shift_div=self.shift_div, shift_place=self.shift_place,
                             fc_lr5=not (self.tune_from and self.dataset in self.tune_from),
                             temporal_pool=self.temporal_pool,
                             non_local=self.non_local)
        self.tsm_model = self.tsm_model.cuda()

        scale_size = self.tsm_model.scale_size
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        normalize = GroupNormalize(input_mean, input_std)
        self.transform_hyj = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
            normalize,
        ])

    def _load_tsm_weights(self, tsm_weights_path):
        if not os.path.exists(tsm_weights_path):
            raise FileNotFoundError(f"TSM模型权重文件路径不存在: {tsm_weights_path}")
        checkpoint = torch.load(tsm_weights_path, weights_only=False)
        self.tsm_model.load_state_dict(checkpoint['state_dict'])
        self.tsm_model.eval()
        logger.info(f"成功加载TSM模型权重: {os.path.basename(tsm_weights_path)}")

    def run(self, frames: list) -> dict:
        if len(frames) != 1:
            raise ValueError("当前版本只支持单帧处理")

        frame = frames[0]
        self.frame_counter += 1

        # TSM动作识别
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img_list = []
        if self.frame_counter % int(25 * 1 / self.num_segments) == 0:
            if len(pil_img_list) >= 8:
                pil_img_list = pil_img_list[1:] + [frame_pil]
            else:
                pil_img_list.append(frame_pil)

            if len(pil_img_list) == 8:
                input_tensor = self.transform_hyj(pil_img_list)
                with torch.no_grad():
                    output = self.tsm_model(input_tensor.unsqueeze(0).cuda())
                state = int(torch.argmax(output).cpu())
                action = self.cls_text[state]
            else:
                action = "未识别"
        else:
            action = "未识别"

        return {
            "frame_id": self.frame_counter,
            "action": action
        }

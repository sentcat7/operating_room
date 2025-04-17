import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from tqdm import tqdm

from stream_infer.algo import BaseAlgo


class TableSeg(BaseAlgo):
    def __init__(self):
        super().__init__(name="Tabel_model")
        self.model = None

    def init(self, model_path: str):
        self._init_model(model_path=model_path)

    def _init_model(self, model_path: str):
        assert os.path.exists(model_path), f"{model_path} is not exist"
        self.model = YOLO(model_path, verbose=False)

    def _get_one_mask(self, data):
        common_mask = None
        for res in data:
            if res.masks is not None:  
                masks = res.masks.data.cpu().numpy()
                # masks = (masks > 0).astype(np.unit8) * 255
                # 遍历每个mask并可视化
                for i, mask in enumerate(masks):
                    # 将mask转换为二值图像
                    mask = (mask > 0).astype(np.uint8) * 255

                    if common_mask is None:
                        common_mask = mask
                    else:
                        common_mask = cv.bitwise_or(common_mask, mask)

        return common_mask
    
    def _get_rect(self, frame, mask):
        if mask is not None:
            mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            largest_contour = max(contours, key=cv.contourArea)
            rect = cv.minAreaRect(largest_contour)
            (center, (width, height), angle) = rect
            box = cv.boxPoints(rect)
            box = np.intp(box)
        else:
            box = None
        return box

    def run(self, frame):
        res = self.model.predict(frame, verbose=False)
        return res[0]
        # mask = self._get_one_mask(res)
        # box = self._get_rect(frame, mask)
        
        # return box

    def test(self, input_video: str, output_path: str = None):
        cap = cv.VideoCapture(input_video)

        if output_path:
            fps = cap.get(cv.CAP_PROP_FPS)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print(f"fps: {fps}, hw: {height}x{width}, cnt:{total_frames}")
            out_cap = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        else:
            out_cap = None

        if not cap.isOpened():
            raise ValueError("cap is not open")
        
        for _ in tqdm(range(total_frames), desc="Processing Video", unit="frame"):
            ret, frame = cap.read()

            if not ret:
                break
            
            box = self.run(frame)
            tmp = frame.copy()
            cv.polylines(tmp, [box], isClosed=True, color=(0, 255, 0), thickness=2)


            if out_cap:
                out_cap.write(tmp)
            else:
                del tmp
        
        if cap:
            cap.release()
        if out_cap:
            out_cap.release()


if __name__ == "__main__":
    x = "/home/test/001-2s.mp4"
    y = "/home/test/001-2s-res.mp4"
    mm = TableSeg("/home/runs/segment/table/weights/best.pt")
    mm.test(
        x, y
    )
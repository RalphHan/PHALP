import os, requests, torch, math
import numpy as np

from yolov6.utils.events import LOGGER
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from phalp.configs.base import CACHE_DIR


class YOLOv6L6:
    def __init__(self):
        checkpoint = os.path.join(CACHE_DIR, "phalp/weights/yolov6l6.pt")
        self.device = torch.device('cuda:0')
        self.model = DetectBackend(checkpoint, device=self.device)
        self.stride = self.model.stride
        self.img_size = 1280
        self.model.model.float()
        self.model.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
            next(self.model.model.parameters())))

    def process_image(self, img_bgr, img_size, stride):
        '''Process image before image inference.'''
        img = img_bgr[..., ::-1]
        img = letterbox(img, img_size, stride=stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0

        return img

    @torch.no_grad()
    def __call__(self, img_src):
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        agnostic_nms = False
        img = self.process_image(img_src, self.img_size, self.stride)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
        pred_results = self.model(img)  # [1, 23800, 85]
        classes = [0]
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        return det

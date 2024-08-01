import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List
from models.yolo.utils import *

class YOLO:
    def __init__(self,
                 weights: str = "yolo.onnx",
                 device: str = "cpu",
                 img_size: Tuple[int, int] = (640, 640),
                 iou_threshold: float = 0.4,
                 score_threshold: float = 0.1,
                 classes: List[str] = None):
        
        self.weights = weights
        self.device = device

        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.classes = classes

        self.session, self.stride, self.names, self.output_names = create_onnx_session(weights, device)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        self.image_height, self.image_width = img.shape[:2]
        img = cv2.resize(img, (self.img_size[1], self.img_size[0])) # Resize using OpenCV
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0) # (1, 3, H, W)
        img = img / 255.0
        img = img.astype(np.float32)

        return img

    def postprocess(self, out):
        out = out[0]

        predictions = np.squeeze(out).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.score_threshold]
        scores = scores[scores > self.score_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # rescale boxes from 0-1 to image dimensions
        boxes = predictions[:, :4] # x1, y1, x2, y2
        input_shape = np.array([self.img_size[1], self.img_size[0], self.img_size[1], self.img_size[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=self.score_threshold, nms_threshold=self.iou_threshold)
        detections = []
        
        for i in indices:
            bbox = xywh2xyxy(np.array([boxes[i]]))[0]
            detections.append({
                "class": class_ids[i],
                "score": scores[i],
                "bbox": bbox,
                "class_name": self.names[class_ids[i]] if self.names else str(class_ids[i])
            })
        
        return detections

    def predict(self, img: np.ndarray) -> np.ndarray:
        x = self.preprocess(img)
        out = self.session.run(self.output_names, {self.session.get_inputs()[0].name: x})
        out = self.postprocess(out)

        return out


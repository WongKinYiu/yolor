from models.models import Darknet
from utils.datasets import letterbox
from utils.general import non_max_suppression
import numpy as np
import torch
import cv2
from collections import defaultdict
class DocumentLayoutDetection():
    def __init__(self, cfg_path, weight, imgsz = 640, auto_size = 64, device = None) -> None:
        #Config device type
        self.imgsz = imgsz
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # print(f"Model is running on {self.device.type}")
        self.model = Darknet(cfg_path, self.imgsz).cuda()
        self.model.load_state_dict(torch.load(weight, map_location=device)['model'])
        self.model.to(device).eval()
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        self.names = [
            'headline',
            'doc',
            'cir_stamp',
            'rec_stamp'
            ]
        print(f"Document detection model loading completed by {self.device.type}")
    
    def relocate_illigal_point(self, point, pmin, pmax):
        point = round(point)
        if point < pmin:
            return pmin
        if point > pmax:
            return pmax
        return point
    

    def single_inference(self, cv2_image, conf_thres = 0.4, iou_thres=0.5):
        detected_objects = None
        with torch.no_grad():
            ori_img = cv2_image
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img_h, img_w, c = ori_img.shape
            img = letterbox(ori_img, new_shape=self.imgsz, auto_size=64)[0]

            # Convert
            scaled_h, scaled_w, c = img.shape
            h_ratio = scaled_h/img_h
            w_ratio = scaled_w/img_w
            
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None) #[[left, top, right, bottom, probs, cls]]
            np_pred = pred[0].cpu().detach().numpy()

            draw_img = ori_img.copy()
            objects = defaultdict(list)
            for each_pred in np_pred:
                relocated_point = []
                left, top, right, bottom, probs, cls = each_pred
                cls = int(cls)
                left = left/w_ratio
                right = right/w_ratio
                top = top/h_ratio
                bottom = bottom/h_ratio

                left = self.relocate_illigal_point(left, 0, img_w)
                top = self.relocate_illigal_point(top, 0, img_h)
                right = self.relocate_illigal_point(right, 0, img_w)
                bottom = self.relocate_illigal_point(bottom, 0, img_h)

                relocated_point = [left, top, right, bottom]
                saving_dict = {
                    "bbox": relocated_point,
                    "score": probs,
                    "cls_name": self.names[cls],
                    "cls_id": cls
                }
                objects[cls].append(saving_dict)
            detected_objects = objects
        return detected_objects

    def __call__(self, cv2_image, conf_thres = 0.4, iou_thres=0.5) -> dict:
        detected_objects = self.single_inference(cv2_image=cv2_image, conf_thres=conf_thres, iou_thres=iou_thres)
        return detected_objects
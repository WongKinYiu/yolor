# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from utils.general import non_max_suppression

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']


def infer_yolor(onnx_path="./weights/yolor-d6-640-640.onnx"):
    ort.set_default_logger_severity(4)
    ort_session = ort.InferenceSession(onnx_path)

    outputs_info = ort_session.get_outputs()
    print("num outputs: ", len(outputs_info))
    print(outputs_info)

    test_path = "./inference/images/horses.jpg"
    save_path = f"./inference/images/horses_{os.path.basename(onnx_path)}.jpg"

    img_bgr = cv2.imread(test_path)
    height, width, _ = img_bgr.shape

    img_rgb = img_bgr[:, :, ::-1]
    img_rgb = cv2.resize(img_rgb, (640, 640))
    img = img_rgb.transpose(2, 0, 1).astype(np.float32)  # (3,640,640) RGB

    img /= 255.0

    img = np.expand_dims(img, 0)
    # [1,num_anchors,num_outputs=2+2+1+nc=cxcy+wh+conf+cls_prob]
    pred = ort_session.run(["output"], input_feed={"images": img})[0]

    print(pred.shape)
    print(pred[0, :4].min())
    print(pred[0, :4].max())
    pred_tensor = torch.from_numpy(pred).float()

    boxes_tensor = non_max_suppression(pred_tensor)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]

    boxes = boxes_tensor.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return
    scale_w = width / 640.
    scale_h = height / 640.

    print(boxes[:2, :])

    boxes[:, 0] *= scale_w
    boxes[:, 1] *= scale_h
    boxes[:, 2] *= scale_w
    boxes[:, 3] *= scale_h

    print(f"detect {boxes.shape[0]} bounding boxes.")

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        print(boxes[i])
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.putText(img_bgr, names[label] + ":{:.2f}".format(conf), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 2)

    cv2.imwrite(save_path, img_bgr)

    print("detect done.")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    infer_yolor(onnx_path="./weights/yolor-p6-640-640.onnx")
    infer_yolor(onnx_path="./weights/yolor-d6-640-640.onnx")
    infer_yolor(onnx_path="./weights/yolor-e6-640-640.onnx")
    infer_yolor(onnx_path="./weights/yolor-w6-640-640.onnx")
    infer_yolor(onnx_path="./weights/yolor-ssss-s2d-640-640.onnx")

    """
    PYTHONPATH=. python3 ./detect_onnx.py
    """

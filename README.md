# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-learn-one-representation-unified/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=you-only-learn-one-representation-unified)

<div align="center">
  <img src="https://github.com/open-mmlab/mmdetection/blob/master/resources/mmdet-logo.png" width="600"/>
</div>

We implement **YOLOR** on famous object detection toolbox [MMDetection](https://github.com/open-mmlab/mmdetection), the results show that **YOLOR** can work well with no matter one-stage/two-stage/multi-stage, anchor-free/anchor-based object detectors and instance segmentors.

#### Supported methods:

- [x] [Faster R-CNN (NeurIPS'2015)](https://arxiv.org/abs/1506.01497)
- [x] [Mask R-CNN (ICCV'2017)](https://arxiv.org/abs/1703.06870)
- [x] [FCOS (ICCV'2019)](https://arxiv.org/abs/1904.01355)
- [x] [ATSS (CVPR'2020)](https://arxiv.org/abs/1912.02424)
- [x] [Sparse R-CNN (CVPR'2021)](https://arxiv.org/abs/2011.12450)

#### ResNet-50 backbone × Standard 1x training schedule

| Model | YOLOR | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> | config |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **[Faster R-CNN](https://arxiv.org/abs/1506.01497)** |  | 37.4% | 58.1% | 40.4% | - | - | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |
| **Faster R-CNN** | :heavy_check_mark: | **37.6%** | **58.5%** | **40.8%** | - | - | - |  |
|  |  |  |  |  |  |  |  |  |
| **[Mask R-CNN](https://arxiv.org/abs/1703.06870)** |  | 38.2% | 58.8% | 41.4% | 34.7% | 55.7% | 37.2% | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) |
| **Mask R-CNN** | :heavy_check_mark: | **38.3%** | **59.1%** | **41.9%** | **34.8%** | **55.8%** | **37.3%** |  |
|  |  |  |  |  |  |  |  |  |
| **[Sparse R-CNN](https://arxiv.org/abs/2011.12450)** |  | 37.9% | 56.0% | **40.5%** | - | - | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py) |
| **Sparse R-CNN** | :heavy_check_mark: | **38.0%** | **56.3%** | **40.5%** | - | - | - |  |
|  |  |  |  |  |  |  |  |  |
| **[FCOS](https://arxiv.org/abs/1904.01355)** |  | **36.6%** | 56.0% | 38.8% | - | - | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py) |
| **FCOS** | :heavy_check_mark: | **36.6%** | **56.1%** | **39.1%** | - | - | - |  |
|  |  |  |  |  |  |  |  |  |
| **[ATSS](https://arxiv.org/abs/1912.02424)** |  | 39.4% | 57.6% | **42.8%** | - | - | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss/atss_r50_fpn_1x_coco.py) |
| **ATSS** | :heavy_check_mark: | **39.6%** | **57.8%** | **42.8%** | - | - | - |  |
|  |  |  |  |  |  |  |  |  |

#### ResNet-50 backbone × Best practice training schedule

| Model | YOLOR | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> | config |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Sparse R-CNN** |  | **45.0%** | 64.1% | 48.9% | - | - | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) |
| **Sparse R-CNN** | :heavy_check_mark: | **45.0%** | **64.3%** | **49.1%** | - | - | - |  |
|  |  |  |  |  |  |  |  |  |

## Citation

```
@article{wang2021you,
  title={You Only Learn One Representation: Unified Network for Multiple Tasks},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2105.04206},
  year={2021}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

</details>

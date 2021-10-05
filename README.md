# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-learn-one-representation-unified/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=you-only-learn-one-representation-unified)

![Unified Network](https://github.com/WongKinYiu/yolor/blob/main/figure/unifued_network.png)

## Backbone

| Model | Backbone | Size | Parameter | FLOPs | Epoch | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  | [ResNet](https://arxiv.org/abs/1512.03385) | 640 | 34.3M | 65.5G | 300 | 47.6% | 65.8% | 51.6% |
|  | [ResNeXt](https://arxiv.org/abs/1611.05431) | 640 | 32.3M | 60.7G | 300 | 47.7% | 66.0% | 52.0% |
|  | [Darknet](https://arxiv.org/abs/1804.02767) | 640 | 52.9M | 120.6G | 300 | 49.5% | 67.9% | 53.7% |
|  | [VoVNet](https://arxiv.org/abs/1904.09730) | 640 |  |  | 300 |  |  |  |
|  | [HarDNet](https://arxiv.org/abs/1909.00948) | 640 |  |  | 300 |  |  |  |
|  | [GhostNet](https://arxiv.org/abs/1911.11907) | 640 | 19.4M | 36.9G | 300 | 45.9% | 64.4% | 49.4% |
|  | [ELAN]() | 640 |  |  | 300 |  |  |  |

## Module

| Model | Module | Size | Parameter | FLOPs | Epoch | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|  | [SE](https://arxiv.org/abs/1709.01507) | 640 |  |  | 30 |  |  |  |
|  | [DeformableConv](https://arxiv.org/abs/1703.06211) | 640 |  |  | 30 |  |  |  |
|  | [NonLocal](https://arxiv.org/abs/1711.07971) | 640 |  |  | 30 |  |  |  |
|  | [CBAM](https://arxiv.org/abs/1807.06521) | 640 |  |  | 30 |  |  |  |
|  | [DeformableConv2](https://arxiv.org/abs/1811.11168) | 640 |  |  | 30 |  |  |  |
|  | [GC](https://arxiv.org/abs/1904.11492) | 640 |  |  | 30 |  |  |  |
|  | [DeformableKernel](https://arxiv.org/abs/1910.02940) | 640 |  |  | 30 |  |  |  |
|  | [SAM](https://arxiv.org/abs/2004.10934) | 640 |  |  | 30 |  |  |  |
|  | [DNL](https://arxiv.org/abs/2006.06668) | 640 |  |  | 30 |  |  |  |
|  | [VisionTransformer](https://arxiv.org/abs/2010.11929) | 640 |  |  | 30 |  |  |  |
|  | [CCN](https://arxiv.org/abs/2010.12138) | 640 |  |  | 30 |  |  |  |
|  | [SAAN](https://arxiv.org/abs/2010.12138) | 640 |  |  | 30 |  |  |  |
|  | [Swin](https://arxiv.org/abs/2103.14030) | 640 |  |  | 30 |  |  |  |
|  | [DyHead](https://arxiv.org/abs/2106.08322) | 640 |  |  | 30 |  |  |  |
|  | [Outllooker](https://arxiv.org/abs/2106.13112) | 640 |  |  | 30 |  |  |  |
|  | [IFM]() | 640 |  |  | 30 |  |  |  |



# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

![Unified Network](https://github.com/WongKinYiu/yolor/blob/main/figure/unifued_network.png)

<img src="https://github.com/WongKinYiu/yolor/blob/main/figure/performance.png" height="480">

Developing... [(weights)](https://github.com/WongKinYiu/yolor/releases/tag/weights)

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOR-P6** | 1280 | **54.1%** | **71.8%** | **59.3%** | **36.2%** | **58.1%** | **65.8%** | 76 *fps* |
| **YOLOR-W6** | 1280 | **55.5%** | **73.2%** | **60.6%** | **37.6%** | **59.5%** | **67.7%** | 66 *fps* |
| **YOLOR-E6** | 1280 | **56.4%** | **74.1%** | **61.6%** | **39.1%** | **60.1%** | **68.2%** | 45 *fps* |
| **YOLOR-D6** | 1280 | **57.3%** | **75.0%** | **62.7%** | **40.4%** | **61.2%** | **69.2%** | 34 *fps* |
|  |  |  |  |  |  |  |

To reproduce the results in the paper, please use this branch.

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOR-P6** | 1280 | **52.6%** | **70.6%** | **57.6%** | **34.7%** | **56.6%** | **64.2%** | 76 *fps* |
| **YOLOR-W6** | 1280 | **54.1%** | **72.0%** | **59.2%** | **36.3%** | **57.9%** | **66.1%** | 66 *fps* |
| **YOLOR-E6** | 1280 | **54.8%** | **72.7%** | **60.0%** | **36.9%** | **58.7%** | **66.9%** | 45 *fps* |
| **YOLOR-D6** | 1280 | **55.4%** | **73.3%** | **60.6%** | **38.0%** | **59.2%** | **67.1%** | 34 *fps* |
|  |  |  |  |  |  |  |
| **YOLOv4-P5** | 896 | **51.8%** | **70.3%** | **56.6%** | **33.4%** | **55.7%** | **63.4%** | 41 *fps* |
| **YOLOv4-P6** | 1280 | **54.5%** | **72.6%** | **59.8%** | **36.6%** | **58.2%** | **65.5%** | 30 *fps* |
| **YOLOv4-P7** | 1536 | **55.5%** | **73.4%** | **60.8%** | **38.4%** | **59.4%** | **67.7%** | 16 *fps* |
|  |  |  |  |  |  |  |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | FLOPs | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOR-P6** | 1280 | **52.5%** | **70.6%** | **57.4%** | **37.4%** | **57.3%** | **65.2%** | 326G | [yolor-p6.pt](https://drive.google.com/file/d/1WyzcN1-I0n8BoeRhi_xVt8C5msqdx_7k/view?usp=sharing) |
| **YOLOR-W6** | 1280 | **54.0%** | **72.1%** | **59.1%** | **38.1%** | **58.8%** | **67.0%** | 454G | [yolor-w6.pt](https://drive.google.com/file/d/1KnkBzNxATKK8AiDXrW_qF-vRNOsICV0B/view?usp=sharing) |
| **YOLOR-E6** | 1280 | **54.6%** | **72.5%** | **59.8%** | **39.9%** | **59.0%** | **67.9%** | 684G | [yolor-e6.pt](https://drive.google.com/file/d/1jVrq8R1TA60XTUEqqljxAPlt0M_MAGC8/view?usp=sharing) |
| **YOLOR-D6** | 1280 | **55.4%** | **73.5%** | **60.6%** | **40.4%** | **60.1%** | **68.7%** | 937G | [yolor-d6.pt](https://drive.google.com/file/d/1WX33ymg_XJLUJdoSf5oUYGHAtpSG2gj8/view?usp=sharing) |
|  |  |  |  |  |  |  |  |
| **YOLOR-S** | 640 | **40.7%** | **59.8%** | **44.2%** | **24.3%** | **45.7%** | **53.6%** | 21G |  |
| **YOLOR-S**<sub>DWT</sub> | 640 | **40.6%** | **59.4%** | **43.8%** | **23.4%** | **45.8%** | **53.4%** | 21G |  |
| **YOLOR-S<sup>2</sup>**<sub>DWT</sub> | 640 | **39.9%** | **58.7%** | **43.3%** | **21.7%** | **44.9%** | **53.4%** | 20G |  |
| **YOLOR-S<sup>3</sup>**<sub>S2D</sub> | 640 | **39.3%** | **58.2%** | **42.4%** | **21.3%** | **44.6%** | **52.6%** | 18G |  |
| **YOLOR-S<sup>3</sup>**<sub>DWT</sub> | 640 | **39.4%** | **58.3%** | **42.5%** | **21.7%** | **44.3%** | **53.0%** | 18G |  |
| **YOLOR-S<sup>4</sup>**<sub>S2D</sub> | 640 | **36.9%** | **55.3%** | **39.7%** | **18.1%** | **41.9%** | **50.4%** | 16G | [weights](https://drive.google.com/file/d/1rFoRk1ZoKvE8kbxAl2ABBy6m9Zl6_k4Y/view?usp=sharing) |
| **YOLOR-S<sup>4</sup>**<sub>DWT</sub> | 640 | **37.0%** | **55.3%** | **39.9%** | **18.4%** | **41.9%** | **51.0%** | 16G | [weights](https://drive.google.com/file/d/1IZ1ix1hwUjEMcCMpl67CHQp98XOjcIsv/view?usp=sharing) |
|  |  |  |  |  |  |  |  |

New training scheme: train 300 epochs only.

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOR-P6** | 1280 | **53.2%** | **70.8%** | **58.2%** | **35.3%** | **57.3%** | **64.6%** | 76 *fps* |
| **YOLOR-W6** | 1280 | **54.7%** | **72.2%** | **59.8%** | **36.7%** | **58.7%** | **66.7%** | 66 *fps* |
| **YOLOR-E6** | 1280 | **55.5%** | **73.1%** | **60.7%** | **37.9%** | **59.2%** | **67.2%** | 45 *fps* |
| **YOLOR-D6** | 1280 | **56.1%** | **73.8%** | **61.4%** | **38.8%** | **59.9%** | **68.4%** | 34 *fps* |
|  |  |  |  |  |  |  |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolor -it -v your_coco_path/:/coco/ -v your_code_path/:/yolor --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
cd /
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .

# go to code folder
cd /yolor
```

</details>

Colab environment
<details><summary> <b>Expand</b> </summary>
  
```
git clone -b paper https://github.com/WongKinYiu/yolor
cd yolor

# pip install required packages
pip install -qr requirements.txt

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install
cd ..

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd ..
```

</details>

Prepare COCO dataset
<details><summary> <b>Expand</b> </summary>

```
cd /yolor
bash scripts/get_coco.sh
```

</details>

Prepare pretrained weight
<details><summary> <b>Expand</b> </summary>

```
cd /yolor
bash scripts/get_pretrain.sh
```

</details>

## Testing

```
python test.py --img 1280 --conf 0.001 --iou 0.65 --batch 32 --device 0 --data data/coco.yaml --weights yolor-p6.pt --name yolor-p6_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52471
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70569
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.37395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.57292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.65187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38975
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.71349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.58020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.75261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.82403
```

## Training

Single GPU training:

```
python train.py --batch-size 8 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --device 0 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
```

Multiple GPU training:

```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 16 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --sync-bn --device 0,1 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
```

Training schedule in the paper:

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights '' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6 --hyp hyp.scratch.1280.yaml --epochs 300
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 tune.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights 'runs/train/yolor-p6/weights/last_298.pt' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6-tune --hyp hyp.finetune.1280.yaml --epochs 450
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg models/yolor-p6.yaml --weights 'runs/train/yolor-p6-tune/weights/epoch_424.pt' --sync-bn --device 0,1,2,3,4,5,6,7 --name yolor-p6-fine --hyp hyp.finetune.1280.yaml --epochs 450
```

## Inference

```
python detect.py --source images/horses.jpg --weights yolor-p6.pt --conf 0.25 --img-size 1280 --device 0
```

You will get the results:

![horses](https://github.com/WongKinYiu/yolor/blob/paper/inference/output/horses.jpg)

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

</details>

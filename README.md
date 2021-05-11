# YOLOR
implementation of paper - [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206)

![Unified Network](https://github.com/WongKinYiu/yolor/blob/main/figure/unifued_network.png)

To reproduce the results in the paper, please use this branch.

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOR-P6** | 1280 | **52.6%** | **70.6%** | **57.6%** | **34.7%** | **56.6%** | **64.2%** | 49 *fps* |
| **YOLOR-W6** | 1280 | **54.1%** | **72.0%** | **59.2%** | **36.3%** | **57.9%** | **66.1%** | 47 *fps* |
| **YOLOR-E6** | 1280 | **54.8%** | **72.7%** | **60.0%** | **36.9%** | **58.7%** | **66.9%** | 37 *fps* |
| **YOLOR-D6** | 1280 | **55.4%** | **73.3%** | **60.6%** | **38.0%** | **59.2%** | **67.1%** | 30 *fps* |
|  |  |  |  |  |  |  |
| **YOLOv4-P5** | 896 | **51.8%** | **70.3%** | **56.6%** | **33.4%** | **55.7%** | **63.4%** | 41 *fps* |
| **YOLOv4-P6** | 1280 | **54.5%** | **72.6%** | **59.8%** | **36.6%** | **58.2%** | **65.5%** | 30 *fps* |
| **YOLOv4-P7** | 1536 | **55.5%** | **73.4%** | **60.8%** | **38.4%** | **59.4%** | **67.7%** | 16 *fps* |
|  |  |  |  |  |  |  |

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | weights |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOR-P6** | 1280 | **52.5%** | **70.6%** | **57.4%** | **37.4%** | **57.3%** | **65.2%** |  |
| **YOLOR-W6** | 1280 | **54.0%** | **72.1%** | **59.1%** | **38.1%** | **58.8%** | **67.0%** |  |
| **YOLOR-E6** | 1280 | **54.6%** | **72.5%** | **59.8%** | **39.9%** | **59.0%** | **67.9%** |  |
| **YOLOR-D6** | 1280 | **55.4%** | **73.5%** | **60.6%** | **40.4%** | **60.1%** | **68.7%** |  |
|  |  |  |  |  |  |  |  |

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
```

You will get the results:

```
```

## Training

Single GPU training:

```
```

Multiple GPU training:

```
```

Training schedule in the paper:

```
```

## Inference

```
```

You will get the results:

```
```

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

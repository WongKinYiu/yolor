## Model Zoo

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| **YOLOv4-CSP** | 640 | **49.1%** | **67.7%** | **53.8%** | **32.1%** | **54.4%** | **63.2%** | 76 *fps* |
| **YOLOR-CSP** | 640 | **49.2%** | **67.6%** | **53.7%** | **32.9%** | **54.4%** | **63.0%** | - |
|  |  |  |  |  |  |  |
| **YOLOv4-CSP-X** | 640 | **50.9%** | **69.3%** | **55.4%** | **35.3%** | **55.8%** | **64.8%** | 53 *fps* |
| **YOLOR-CSP-X** | 640 | **51.1%** | **69.6%** | **55.7%** | **35.7%** | **56.0%** | **65.2%** | - |
|  |  |  |  |  |  |  |

## Installation

https://github.com/AlexeyAB/darknet

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

```
# get code
git clone https://github.com/AlexeyAB/darknet

# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolor -it -v your_coco_path/:/coco/ -v your_code_path/:/yolor --shm-size=64g nvcr.io/nvidia/pytorch:21.02-py3

# apt install required packages
apt update
apt install -y libopencv-dev

# edit Makefile
#GPU=1
#CUDNN=1
#CUDNN_HALF=1
#OPENCV=1
#AVX=1
#OPENMP=1
#LIBSO=1
#ZED_CAMERA=0
#ZED_CAMERA_v2_8=0
#
#USE_CPP=0
#DEBUG=0
#
#ARCH= -gencode arch=compute_52,code=[sm_70,compute_70] \
#      -gencode arch=compute_61,code=[sm_75,compute_75] \
#      -gencode arch=compute_61,code=[sm_80,compute_80] \
#      -gencode arch=compute_61,code=[sm_86,compute_86]
#
#...

# build
make -j8
```

</details>

## Testing

To reproduce inference speed, using:

```
CUDA_VISIBLE_DEVICES=0 ./darknet detector demo cfg/coco.data cfg/yolov4-csp.cfg weights/yolov4-csp.weights source/test.mp4 -dont_show -benchmark 
```

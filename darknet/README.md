## Model Zoo

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv4-CSP** | 640 | 48.8% | **67.5%** | 53.1% | **33.1%** | 54.1% | 62.3% | 76 *fps* |
| **YOLOR-CSP** | 640 | **49.1%** | **67.5%** | **53.5%** | 32.5% | **54.3%** | **62.6%** | - |
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

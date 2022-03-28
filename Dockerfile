FROM nvcr.io/nvidia/pytorch:20.11-py3

LABEL maintainer="Haneol Kim <bestwook7@gmail.com>"

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx
RUN pip install seaborn thop

RUN cd /
RUN git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python setup.py build install

RUN cd / && git clone https://github.com/fbcotter/pytorch_wavelets && cd pytorch_wavelets && pip install .

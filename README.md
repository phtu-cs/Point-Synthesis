# Point Pattern Synthesis via Irregular Convolution
by [Peihan Tu](https://scholar.google.com/citations?user=UA-ENWAAAAAJ&hl=en), [Dani Lischinski](http://danix3d.droppages.com/), [Hui Huang](https://vcc.tech/~huihuang)

## Project page
The project page is available at [https://vcc.tech/research/2019/PointSyn](https://vcc.tech/research/2019/PointSyn).

## Introduction ##
This repository contains an implementation for [Point Pattern Synthesis via Irregular Convolution](https://vcc.tech/file/upload_file//image/research/att201908150924/PointSyn.pdf). 
This method takes an input point pattern and generate visually similar output point pattern by optimization with a neural network.
The implementation is in Python and [Pytorch](https://pytorch.org/) and Matlab.

You will need CUDA-compatible GPUs for generating results within several minutes.

If you have questions, please feel free to contact Peihan Tu (phtu@cs.umd.edu).

![overview](overview.jpg)

## Starting ##

The current released codes are tested on Ubuntu 16.04. To train this network properly, please install the follow dependencies:
- Python 3.6
- CUDA 8.0
- Cudnn 6.0
- Pytorch
- Numpy

Clone our repo
```
git clone https://github.com/tph9608/Point-Synthesis.git
```

## Run

You can simply generate results shown in the paper by runing
```bash
./exp.sh
```
Please go into the results_final folder the check all of the results.

## Cite ##

If you use our code, please cite our paper:
```
@article{PointSyn19,
title = {Point Pattern Synthesis via Irregular Convolution},
author = {Peihan Tu, Dani Lischinski and Hui Huang},
journal = {Computer Graphics Forum (Proceedings of SGP 2019)},
volume = {38},
number = {5},
year = {2019},
} 

```
## License ##
MIT License

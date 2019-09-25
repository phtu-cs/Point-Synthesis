# Point Pattern Synthesis via Irregular Convolution
by [Peihan Tu](https://scholar.google.com/citations?user=UA-ENWAAAAAJ&hl=en), [Dani Lischinski](http://danix3d.droppages.com/), [Hui Huang](https://vcc.tech/~huihuang)

## Project page
The project page is available at [https://vcc.tech/research/2019/PointSyn](https://vcc.tech/research/2019/PointSyn).

## Introduction ##
This repository contains an official implementation for [Point Pattern Synthesis via Irregular Convolution](https://vcc.tech/file/upload_file//image/research/att201908150924/PointSyn.pdf). 
This method takes an input point pattern and generate visually similar output point pattern by optimization with a neural network.
The implementation is in Python and [Pytorch](https://pytorch.org/) and Matlab.

You will need CUDA-compatible GPUs for generating results within several minutes.

If you have questions, please feel free to contact Peihan Tu (phtu@cs.umd.edu).

## Prerequisite

- Pytorch
- Matlab (make sure matlab is in your Path)

## Run

```bash
./exp.sh
```

## SCAN: Semi-supervisedly Co-embedding Attributed Networks

This repository contains the Python implementation for SCAN. Further details about SCAN can be found in our paper:

> Zaiqiao Meng, Shangsong Liang, Jinyuan Fang, Teng Xiao. Semi-supervisedly Co-embedding Attributed Networks. (NeurIPS 2019)

- A pytorch implementation can be found [here](https://github.com/GuanZhengChen/SCAN-Pytorch)


## Introduction

SCAN is a semi-supervised co-embedding model for attributed networks based on the generalized SVAE for heterogeneous data, which collaboratively learns low-dimensional vector representations of both nodes and attributes for partially labelled attributed networks semi-supervisedly. The node and attribute embeddings obtained in a unified manner by our SCAN can benefit not only for capturing the proximities between nodes but also the affinities between nodes and attributes. Moreover, our model also trains a discriminative network to learn the label predictive distribution of nodes.

## Requirements

=================

- TensorFlow (1.0 or later)
- python 3.6
- scikit-learn
- scipy

## Run the demo

=================

```
python main.py
```

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{meng2019scan,
  title={Semi-supervisedly Co-embedding Attributed Networks},
  author={Meng, Zaiqiao and Liang, Shangsong and Fang, Jinyuan and Xiao, Teng},
  booktitle={Advances in neural information processing systems},
  year={2019}
}
```

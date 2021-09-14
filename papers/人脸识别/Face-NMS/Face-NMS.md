# Face-NMS

## Abstract
人脸识别模型的训练数据的规模越来越大，所需的时间和算力也越来越多，本文提出通过解决由贪婪采样操作（即核心组选择视角？core-set selection perspective）引起的数据集的冗余问题。提出一个全新的过滤策略称为Face-NMS.Face-NMS运行于特征空间，在生成core sets的时候同时兼顾局部和全局的稀疏性。实际上，Face-NMS与物体检测中的NMS很相近。它根据人脸对全局稀疏性的潜在贡献进行排名，并过滤出在局部稀疏性上有高相似度的多余的人脸。

## 1. Introduction
Face-NMS生成的core sets在相对高稀疏的情况下更合理，如Figure 1所示。
![Figure 1](1.png "Figure 1")

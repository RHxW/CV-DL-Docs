# Memory-Based Neighbourhood Embedding for Visual Recognition

## Abstract
为了获得更优的特征嵌入，一些方法通过设计不同的网络或损失函数来实现，本文提出通过基于内存的邻接嵌入(MNE, Memory-based Neighbourhood Embedding)通过考虑其相邻特征来增强一个普通的CNN特征。

## 1. Introduction
通过网络获取的特征是基于单张图像的，并忽略了丰富的上下文信息。当同一类别物体的差异变化剧烈或者不同类别物体外观差异很小的时候，就难以根据单张图片来分辨类别。如Figure 1所示，两个类别的特征距离很近。但是如果将个别特征和其相邻特征考虑在内，就可以将两个不同的特征分成两个聚类，而它们的邻接关系也可以用于修改原始特征得到分辨能力更强的新特征。

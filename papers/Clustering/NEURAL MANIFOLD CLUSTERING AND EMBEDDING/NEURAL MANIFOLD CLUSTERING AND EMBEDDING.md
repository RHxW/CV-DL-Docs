# NEURAL MANIFOLD CLUSTERING AND EMBEDDING

## Abstract
给定一组非线性流形，非线性子空间聚类或者流形聚类的目标是对流形结构上的数据点进行聚类，同时学习将每个流形作为一个特征空间的线性子空间进行参数化。深度神经网络因为其在容量和弹性上的优势，潜在地适合于在高度非线性情况下进行这种操作。我们主张，要达成用神经网络进行流形聚类需要两个基本要素：一个domain-specific的约束，用来保证流形的身份；以及一个学习算法，用来将每个流形映射到特征空间的一个线性子空间。本work展示了很多约束都可以通过数据增强来实现。对于子空间特征学习，可以用Maximum Coding Rate Reduction ($\text{MCR}^2$)作为目标函数。结合在一起就得到了*Neural Manifold Clustering and Embedding*(NMCE)，是一个用于通用场景流形聚类的新方法，它的性能大幅度超越了基于自编码器的深度子空间聚类方法。而且在更具挑战性的自然图片数据集上，NMCE也能够超越其他专门为聚类设计的算法。定性地说，我们展示了NMCE学习到了一个有意义且可解释的特征空间。因为NMCE的公式和一些重要的自监督学习(Self-supervised learning, SSL)方法联系紧密，我们相信这个工作可以帮助更进一步了解SSL表示学习。

## 1 Introduction
我们考察了无监督表示学习的方法，它的目标是在不使用任何标签的情况下从数据中学习到结构（特征）信息。如果数据分布于一个线性子空间，那么这个线性子空间可以通过PCA算法提取得到，PCA算法是无监督学习的一个最基础的形式。当数据占据了几个线性子空间的并集，则需要子空间聚类来将每个数据点聚类到一个子空间同时估计每个子空间的参数。在这里我们要考虑的是更有挑战性场景：当数据点来自于多个非线性低维流形的组合。在这种场景下，聚类问题可以表示为：

**Task 1.** Manifold Clustering and Embedding:给定来自于多个非线性低维流形组合的数据点，要根据它们对应的流形对它们进行划分，并且获取每个流形的一个低维embedding

虽然针对这个问题已经提出了一些解决方案，但是如何有效地使用神经网络解决流形聚类问题仍然是一个开放的问题。本文提出的*Neural Manifold Clustering and Embedding*(NMCE)遵循以下三个原则：

1. 聚类和表达应该遵循一个domain-specific的约束，例如局部邻居、局部线性插值或数据增强不变性。
2. 一个特定流形的embedding不会坍塌
3. 确定流形的embedding应该是线性可分离的，亦即它们由不同的线性子空间构成

通过使用数据增强实现1，通过使用子空间特征学习算法Maximum Coding Rate Reduction ($\text{MCR}^2$)实现2和3。

本文的主要贡献如下：

1. 将数据增强和$\text{MCR}^2$结合在一起得到一个用于通用目的流形聚类嵌入的新算法NMCE，也讨论了这个算法和自监督对比学习之间的联系
2. 展示了NMCE在标准子空间聚类基准上的性能表现，超越了最好的聚类算法。

## 2 Related Work
**Manifold Learning.** 经典流形学习的目标是将流形结构的数据映射到一个低维表达空间并保留流形结构。有两个主要因素：
1. 保留原空间的一个几何性质
2. embedding不应该坍塌成平凡解

**Manifold Clustering and Embedding.** 

**Self-Supervised Representation Learning.**

**Clustering with Data Augmentation.**

## 3 Neural Manifold Clustering and Embedding
### 3.1 Problem Setup

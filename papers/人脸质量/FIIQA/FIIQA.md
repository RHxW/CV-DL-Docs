# Illumination Quality Assessment for Face Images: A Benchmark and a Convolutional Neural Networks Based Model

**Abstract** 评估人脸图片的光照质量，两件事：1. 建立了一个人脸光照数据集(FIIQD, Face Image Illumination Quality Database)，内部包含224733张不同光照模式的人脸图片，而且每张图片对应一个光照质量评分。2. 提出一个基于深度卷积网络的新模型用来预测人脸图片的光照质量。



## 1 Introduction

目标是要设计一个算法，其可以自动且高效地评估一张给定人脸图片的光照质量，而且评估结果应该和人类的判断相关。

![Figure 1](1.png"Figure 1")



### 1.1 Related Work

blahblahblah

实际上，FIIQA问题可以看作一种特殊的NR-IQA（无参考图片质量评估）问题。

blahblahblah



### 1.2 Our Motivations and Contributions

blahblahblah

两个贡献：大规模数据集和网络模型

blahblahblah



## 2 FIIQD: A Face Image Illumination Quality Database

使用半自动策略构建FIIQD数据集。主要有四步，构建原始光照模式图片集，光照模式图片的主观评价，构建目标脸集合，光照变换。pipeline如Fig 2所示

![Figure 2](2.png"Figure 2")



### 2.1 Step 1: The Construction of the Image Set with Source Illumination Patterns

这一步的目的是获取包含真实场景不同光照模式的人脸图片集。这些图片应该能够提供原始光照模式以用于对目标图片进行变换，所以应该包含足够多的光照情况。

![Table 1](t1.png"Table 1")

如Table 1所示，选择了6种场景以覆盖最常见的应用场景。对每个场景，通过组合不同光照模式和时间点定义多个不同的光照情况。对每一种光照情况拍摄至少16张图片（每45°两张）。最后收集了不同原始光照模式的共499张图片并选取了其中的200张高质量图片组成带光照模式的图片集$\mathcal{R}$. Fig 3展示了图片集中的6个样本。

![Figure 3](3.png"Figure 3")

### 2.2 Step 2: Subjective Evaluation of Images in $\mathcal{R}$

这一步中使用主观评定的方式对图片集中图片的光照质量进行评分。采用一种single-stimulus连续质量评估方式进行评分呢。然后对原始评分进行一些后处理。首先过滤出强偏差的主管评分，满足：
$$
d_{ij}-\bar{d_{j}}>T\cdot \sigma_j \qquad (1)
$$
其中$d_{ij}$是第$i$个评估者给出的图片$\textbf{R}_j\in \mathcal{R}$的光照质量评分，$\bar{d_j}$是$\textbf{R}_j$的平均分，$T$是常数阈值，$\sigma_j$是$\textbf{R}_j$评分的标准差。然后，为了消除不同评估者采用的主观标准不同而产生的影响，将原始评分$d_{ij}$转换成
$$
z_{ij}=\frac{d_{ij}-\bar{d_i}}{\sigma_i} \qquad (2)
$$
其中$\bar{d_i}$和$\sigma_i$是第$i$个评估者在$\mathcal{R}$上全部图片评分的均值和标准差。将$\textbf{R}_j$的平均评分作为其最终的主观光照质量评分
$$
s_j=\frac{1}{N_j}\sum z_{ij} \qquad (3)
$$
其中$N_j$是$\textbf{R}_j$的主观评分数量。

至此，对每张图片$\textbf{R}_j\in \mathcal{R}$都获取了反映其光照质量的主观评分$s_j$.

### 2.3 Step 3: Target Face Set Construction

这一步中，选择1014个人（从YaleB，PIE，FERET等）在相同光照情况下的1134张图片构建目标人脸图片集。考虑到多样性，目标人脸图片集中包含多个属性（如脸型、种族、肤色、性别、年龄等）的广泛分布。

### 2.4 Step 4: Illumination Transfer

这一步中，使用光照变换算法（<Face Illumination Transfer through Edge-preserving Filters>）将$\mathcal{R}$中的光照模式变换到目标集合。生成的图片就是FIIQD的图片，见Fig 4所示。

![Figure 4](4.png"Figure 4")

假设$\textbf{R}_j\in \mathcal{R}$是一张带原始光照模式的图片，其主观评分为$s_j$.则根据它生成的图片的评分与它一致。



## 3 $FIIQA_{DCNN}$: A Face Illumination Quality Assessment Model Based on DCNN

网络使用Resnet-50，最后接200个分类，因为FIIQD数据集中的光照模式有200种。
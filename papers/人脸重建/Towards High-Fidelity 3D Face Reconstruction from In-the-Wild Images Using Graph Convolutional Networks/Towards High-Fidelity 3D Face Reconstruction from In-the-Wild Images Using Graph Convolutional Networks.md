# Towards High-Fidelity 3D Face Reconstruction from In-the-Wild Images Using Graph Convolutional Networks

## Abstract
3DMM方法中涉及的面部纹理复原方法有失真的问题。最近的方法会在大规模的高清UV图上训练一个生成网络来解决这一问题，但是这种方法难以实施。本文介绍一个用从单一图片提取的高清纹理重建3D人脸的方法，而不需大规模的人脸问题数据库。核心思想是根据输入图片的面部细节优化3DMM生成的初始纹理。为此，提出使用图卷积网络来重建mesh顶点的颜色，而非重建UV图。

## 1. Introduction
尽管3DMM方法得到的纹理缺少细节信息，但仍能提供整个人脸全局的合理的颜色。也就可以在这个初始的纹理基础上进行优化。具体使用GCN对图片特征进行解码并将细节RGB值传播到人脸mesh上。
我们的重建框架采用由粗到细的方法，基于3DMM模型和GCN. 训练一个CNN来回归3DMM系数（id、表情、纹理）并根据2D图片渲染参数（姿态、光照）。通过3DMM模型，可以得到面部形状和初始的纹理信息。然后关键一步，使用一个预训练好的CNN来提取面部特征，并传入一个GCN来生成顶点的精细颜色。采用一个微分渲染层来实现自监督训练，并通过GAN loss来提升效果。

## 2. Related Work

## 3. Approach
![Figure 2](2.png "Figure 2")
提出的coarse-to-fine方法如Figure 2所示，框架由3个模块组成。特征提取模块包含一个用于回归3DMM系数、面部姿态、和光照参数的回归器以及一个用来提取特征的FaceNet模型。纹理优化模块由三个GCN组成：一个GCN解码器用于对FaceNet提取的特征进行解码并为mesh顶点生成精细颜色；一个GCN优化器用于优化回归器生成的顶点颜色；以及一个合并器将两个结果进行合并得到最终的顶点颜色。判别器会通过对抗训练的方式提升优化效果。

### 3.1. 3DMM Coefficients Regression

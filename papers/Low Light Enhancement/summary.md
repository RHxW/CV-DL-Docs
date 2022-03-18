# Low-Light Enhancement

## 1. Introduction

## 2. Methods
### 2.1 Retinex Theory
Retinex理论认为一张彩色图像可以分解成反射(reflectance)和光照(illumination)两部分，而reflectance属于图像中的固定属性，不随光照的变化而变化，因此只需要对illumination部分进行调整即可实现光照增强效果。

根据Retinex理论，光照增强任务可以分为两个主要步骤（子任务）：图像分解和光照增强。
### 2.2 Dehaze Methods
认为低光照图像的反转图像与有雾图像性质相似，对反转图进行去雾操作可以实现光照增强的效果。

该方向没有物理意义支撑

### 2.3 Curve Estimation
受图像处理软件中调整图像曲线的功能所启发，利用深度神经网络对调整曲线的参数进行估计，利用估计所得曲线对低光照图片进行调整实现光照增强效果，如Zero-DCE
# Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation

## Abstract
本文提出一个称为Zero-Reference Deep Curve Estimation(Zero-DCE)的图像增强方法，将图像增强视为使用深度神经网络实现的单张图像曲线估计任务。本方法训练一个轻量级网络DCE-Net用来估计给定图像的像素级和高阶动态调整曲线。曲线估计的设计考虑到了像素值的范围、单调性和可微性。Zero-DCE的特点在于它不需要任何参考，也就是说训练的时候不需要成对的数据，而这个特点是通过一系列无参考loss函数实现的。

## 1. Introduction

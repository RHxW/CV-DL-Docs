# Partial FC
## Abstract

## Introduction
contributions：
1. 提出一个softmax的近似算法，可以只使用10%的类数据而保证精度不降低
2. 提出一个分布式训练策略
3. Glint360k数据集

## Method
### Problem Formulation
**Model parallel** 在使用大量id进行人脸识别模型训练的时候，受单张显卡显存的限制，如果不使用模型并行那么训练过程会很痛苦。
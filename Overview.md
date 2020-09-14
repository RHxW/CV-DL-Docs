# Overview

* 3DDFA：TODO
* 2DASL：TODO
* PRNet：TODO
* WingLoss：为了解决关键点定位中关键点位置回归loss对小误差不敏感的特点而设计的loss函数
* AdaptiveWingLoss：Todo
* CornerNet：Todo
* DAN：多阶段人脸关键点检测方法，上一个阶段的输出作为下一个阶段的输入，通过多个阶段迭代优化热力图
* LAB：人脸关键点是高度结构化的数据，每个关键点都和一个明确定义的边界相关，所以可以利用面部结构的边界信息（线条）来辅助关键点回归，生成的边界质量越高，关键点回归就越准。
* RetinaFace：检测+对齐+dense回归（图卷积网络）
* SSR-Net：将年龄估计问题用多分类+分类结果回归的方式解决
* FSA-Net：基于SSR-Net的姿态估计方法，加入注意力图
* FocalLoss：提出FocalLoss解决类间不平衡问题，并提出了一个一阶段检测网络RetinaNet
* SSH：一阶段人脸检测，在不同尺度的特征图上使用检测模块实现多尺度检测
* PFLD：关键点检测，辅助网络估计旋转信息用于辅助关键点回归
* Deformable Convolutional Networks：针对传统卷积中固定的几何结构（卷积核）在面对几何变换建模上的天生不足而提出形变卷积和形变RoI pooling用于加强CNN对形变的建模能力。实现方式是通过引入一个特征图上的偏移量来改变感受野区域。
* SSD：Todo
* HR-Net：高分辨率网络，能在整个流程中保持高分辨率表示，用于人体姿态估计，也可做新的backbone
* FAB：一个在模糊视频中利用时间维度上结构一致性的面部关键点检测的框架
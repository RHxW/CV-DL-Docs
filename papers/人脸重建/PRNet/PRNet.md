# Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network

设计了一个2D表示称为UV位置图，在UV空间记录面部的3D形状，训练一个简单的CNN从2D图片回归得到UV位置图。还在loss函数中引入一个加权掩码来提升性能。

## 1 Introduction
提出一个端到端的方法用来联合预测密度对齐和3D人脸形状的重建，方法称为位置图回归网络(Position map Regression Network, PRN)。我们设计了一个UV位置图，是一张2D的图片，其中记录了一组完整的面部点云的3D坐标，同时在每个UV位置上保留了语义信息。然后用一个加权loss训练一个简单的编码解码网络，更关注通过一张2D图片回归得到UV位置图的判别区域。

## 2 Related Works

## 3 Proposed Method
### 3.1 3D Face Representation
目标是通过一张2D图片回归出3D面部几何结构以及对应的密度信息。因此需要一个可以直接通过深度网络预测的表示。一个简单的想法是将全部3D人脸点的坐标组成一个向量并用一个网络进行预测。但是这种变化增加了训练的难度，因为3D空间到1D向量的映射忽略了点与点之间的空间连接信息。
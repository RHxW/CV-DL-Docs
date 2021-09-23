# Sample and Computation Redistribution for Efficient Face Detection

介绍两个简单有效的方法：
1. 采样重分布(SR, Sample Redistribution)，基于基准数据集的统计数据，对训练样本在最需要的阶段进行增强
2. 计算重分布(CR, Computation Redistribution)，使用仔细定义的搜索方法，在backbone, neck和head之间重新分配计算量

通用目标检测的网络结构（包括backbone, neck和head）对于人脸检测的任务来说并不是最优的，之前也有一些方法对人脸检测的网络结构进行了一些探索，也取得了一些成果。尽管这些工作发现了直接将通用的backbone, neck和head用于人脸检测的局限性，但是CR-NAS只关注了backbone的优化，BFbox忽略了对head的优化，ASFD只探索了neck部分的最佳设计。
在固定分辨率（VGA分辨率640*480）上探索高效的人脸检测方法。

contributions
1. 探索了VGA分辨率下的高效人脸检测算法，提出一个采样重分布策略帮助在shallow阶段获取更多训练样本
2. 设计了简单化的搜索空间（神经网络搜索），用于人脸检测器的网络不同部分（backbone, neck和head）间的计算重分布。提出的二阶段计算重分布方法能够轻易得到计算量分布的理解
3. 做实验

因为固定了较小的分辨率（640*480），所以大多数人脸都会在stride 8的特征图上预测。因此首先调查了不同尺度特征图上正样本的重分布。然后探索了不同尺度特征图以及不同网络部分上的计算重分布

### Sample Reallocation
发现stride为8的特征图最重要，当尺寸设定为640像素的时候，大多数的人脸都小于32*32
训练数据的增广，一般是方形patches，尺寸是原图短边的[0.3,1]，现在改成原图短边的[0.3,2]，超出部分用RGB均值填充。

### Computation Redistribution
直接在一个固定尺寸的人脸检测任务上使用分类网络的backbone可能不是最优的。因此采用了网络架构搜索将计算量在backbone, neck和head间进行重分配。
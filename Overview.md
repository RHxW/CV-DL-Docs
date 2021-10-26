# Overview

* AdaptiveWingLoss：一个损失函数，在L1和L2基础上进行了修改，目的是更关注较小的损失
* Attention Is All You Need：Transformers的文章，encoder-decoder的自注意力结构
* Bilinear CNNs：TODO
* ByeGlassesGAN：去眼镜、加眼镜的GAN
* Clustering：
  * CDP：共识传播（集成学习）
  * Density-Aware Feature Embedding for Face Clustering：密度链获取节点特征（密度概念与gcnv中节点置信度很相似）
  * VEGCN：图卷积用于图结构做聚类；v:顶点，e:边
  * Efficient Large-Scale Face Clustering Using an Online Mixture of Gaussians：混合高斯模型在线聚类，具体是认为只用一个聚类无法较好覆盖一个人的人脸图片分布，因此采用多个聚类来代表一个人，于是设计了一套规则来判断某一聚类是一个人的子聚类还是主聚类，另外还设计了一套在线聚类的流程，较繁琐
  * Online Deep Clustering for Unsupervised Representation Learning：用于人脸识别模型训练的半监督在线聚类
  * Structure-Aware Face Clustering on a Large-Scale Graph with 10^7 Nodes：STAR-FC，用于大规模数据训练的GCN聚类方法；由于GCN的训练需要把整张图传入模型，这就导致训练数据的规模受显存限制。可以通过随机采样的方式将子图传入模型训练，但是这样无法保证子图包含全局的结构信息。提出一个能同时覆盖局部结构信息和全局结构信息的随机采样法和一套剪枝方案

* CornerNet：TODO
* DAN：多阶段人脸关键点检测方法，上一个阶段的输出作为下一个阶段的输入，通过多个阶段迭代优化热力图
* Deep Multi-Center Learning for Face Alignment：将人脸划分为多个子区域分别进行关键点定位，再和并得到最终定位结果
* DEEP VARIATIONAL INFORMATION BOTTLENECK：变分信息瓶颈，信息瓶颈理论认为神经网络可以提供一个瓶颈用于压缩映射到隐空间的数据量，去掉的是无价值数据或噪声数据；变分信息瓶颈是信息瓶颈理论的变分法近似，具体实现是将隐空间分布（通过KL散度）拉向标准正态分布（在目标函数中加入这个正则化项），以期提升泛化性能
* Deformable Convolutional Networks：针对传统卷积中固定的几何结构（卷积核）在面对几何变换建模上的天生不足而提出形变卷积和形变RoI pooling用于加强CNN对形变的建模能力。实现方式是通过引入一个特征图上的偏移量来改变感受野区域。
* LAB：人脸关键点是高度结构化的数据，每个关键点都和一个明确定义的边界相关，所以可以利用面部结构的边界信息（线条）来辅助关键点回归，生成的边界质量越高，关键点回归就越准。

* FAB：一个在模糊视频中利用时间维度上结构一致性的面部关键点检测的框架
* FocalLoss：提出FocalLoss解决类间不平衡问题，并提出了一个一阶段检测网络RetinaNet
* FSA-Net：基于SSR-Net的姿态估计方法，加入注意力图
* GAN：
  * PG-GAN：渐进式生成架构，可以生成高清图片
  * StyleGAN：通过映射网络得到style向量，在基于PG-GAN的架构上，每一层都引入style控制向量和随机向量，从而生成特征分离度更高的人脸图片
  * DiscoFaceGAN：引入人脸3DMM的先验，使模型对姿态、表情、光照的分离效果更好
  * DyStyle：为每一个属性引入一个expert网络做隐空间向量的映射，以期得到更优的分离效果
  
* GCN：
  * Semi-Supervised Classification with Graph Convolutional Networks：17年图卷积论文，原理和实现形式

* GNN:
  * Inductive Representation Learning on Large Graphs: GraphSAGE，inductive方法，提出几个聚合器（平均、池化、LSTM）；与基于矩阵分解的嵌入方法不同，GraphSAGE利用节点特征（如文本属性、节点描述信息、节点度）来学习一个能够泛化到没见过节点的嵌入函数。通过在学习算法中结合节点特征，同时学习每个节点邻居的拓扑结构，以及节点特征在邻居中的分布。
  * GRAPH ATTENTION NETWORKS: TODO
  
* HR-Net：高分辨率网络，能在整个流程中保持高分辨率表示，用于人体姿态估计，也可做新的backbone
* Inception：TODO
* LAB：边界感知人脸对齐算法，使用热力图辅助
* MobileNet：
  * MobileNet V1：深度可分离卷积（逐层+逐点）
  * MobileNet V2：TODO
* Objects as Points：Center net
* PFLD：关键点检测，辅助网络估计旋转信息用于辅助关键点回归
* SSD：Todo
* SSR-Net：将年龄估计问题用多分类+分类结果回归的方式解决
* Synthesizing Normalized Faces from Facial Identity Features：从特征重构归一化人脸
* Visualizing and Understanding Convolutional Networks：TODO
* WingLoss：为了解决关键点定位中关键点位置回归loss对小误差不敏感的特点而设计的loss函数
* YOLO：
  * YOLO V1：TODO
* 超分
* 去模糊：
  * Online Video Deblurring via Dynamic Temporal Blending Network：视频去模糊，动态时域混合网络
  * Unsupervised Domain-Specific Deblurring via Disentangled Representations：将模糊信息从图像中解耦

* 人脸检测
  * DSFD：TODO
  * RetinaFace：检测+对齐+dense回归（图卷积网络）
  * SCRFD：检测模型，神经架构搜索
  * SSH：一阶段人脸检测，在不同尺度的特征图上使用检测模块实现多尺度检测
  * YOLO5Face：用YOLOv5进行人脸检测

* 人脸识别
  * ArcFace：基于softmax loss，引入angular margin
  * GroupFace：引入人脸特征用来辅助识别，每个属性对应一个映射网络，得到一个对应属性特征，然后将不同属性特征根据预测的概率进行加权得到最终的人脸表达；group是自动生成的（无监督方式）
  * MagFace：将人脸质量信息融入到人脸特征中（以向量长度反映）
  * VirFace：TODO
  * Variational Prototype Learning：TODO

* 人脸质量
  * Deep Tiny Network for Recognition-Oriented Face：
  * EQFace
  * Face image quality assessment
  * FaceQnet
    * v0
    * v1
  * Inducing Predictive Uncertainty Estimation for Face Recognition
  * IQA
    * RankIQA
  * QAN
    * QAN
    * QAN++
  * SDD-FIQA
  * SER-FIQ

* 人脸重建
  * 3DDFA：
    * v1：
    * v2：
  * 2DASL：
  * PRNet：
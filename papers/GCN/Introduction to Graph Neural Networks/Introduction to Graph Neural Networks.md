# Introduction to Graph Neural Networks

# Chapter 1. Introduction

图是一种对一组对象（节点）和他们之间的关系（边）进行建模的数据结构。作为机器学习的一种独特的数据结构，图在节点分类、连接预测和聚类上吸引了注意力。图神经网络（GNNs）是在图域上操作的基于深度学习的方法。由于其有说服力的效果和高度的可解释性，GNN成为了广泛应用于图分析的方法。

## 1.1 Motivations

### 1.1.1 Convolutional Neural Networks

首先GNNs受到了CNNs的影响。对于高表达能力的特征，CNNs能够提取并构成多尺度局部空间特征，在几乎所有机器学习领域都取得了breakthroughs并在深度学习领域掀起一场革命。深入CNNs和图，发现了CNNs的关键：局部连接，共享权重和多层的使用。这些关键点同样在解决图域上的问题中有着重要地位，因为（1）图是最典型的局部连接结构，（2）和传统的谱图理论相比，权重的共享减少了计算量，（3）多层结构是解决层级模式的关键，它可以捕捉不同尺寸的特征。

但是CNNs只能处理传统的欧几里得数据，例如图片（2D grid）和文本（1D sequence），同样也可以被看作是图的实例。因此寻找CNNs在图上的推广就很好理解了。如Figure 1.1所示，难以定义局部卷积滤波器和池化操作，这阻碍了CNN从欧氏域向非欧域的变换。

![Figure1.1](1.1.png"Figure 1.1")

### 1.1.2 Network Embedding

另一个动机来自于图embedding，其学习使用低维向量来表示图的节点、边或者子图。在图的分析中，传统机器学习方法通常依赖于人工设计的特征且受限于其自身的不可扩展性和高成本。这一类方法的缺点主要有两个：1. 编码器节点间不共享参数，也就代表着参数数量随着节点数量线性增长，导致计算成本很高。2. 直接embedding的方法缺少泛化能力，意味着不能解决动态图或推广到新图上。



# Chapter 1. Basics of Math and Graph

## 2.1 Linear Algebra



...

...

...





# Chapter 4. Vanilla Graph Neural Networks

## 4.1 Introduction

在图中，自然会用节点的特征和与之相关的节点来定义它。GNN的目的是学习一个状态embedding $\textbf{h}_v\in \mathbb{R}^s$，它对每个节点的邻居信息进行了编码。状态embedding $\textbf{h}_v\in \mathbb{R}^s$用于生成一个输出$\textbf{o}_v$，例如预测的节点标签的分布。

![Figure4.1](4.1.png"Figure 4.1")

对于Scarselli的文章，Figure 4.1 展示了一个典型的图。原始的GNN模型解决无向齐次图，其中每个节点都有输入特征$\textbf{x}_v$且每条边也可以有其特征。这篇文章使用$co[v]$和$ne[v]$来表示$v$的边和邻居节点的集合。

## 4.2 Model

给定节点的输入特征和边，下面讨论一下模型如何获取节点的embedding $\textbf{h}_v$和输出embedding $\textbf{o}_v$

为了根据输入邻居来更新节点状态，用到一个参数函数$f$，称为局部转移函数，由所有节点共享。为了生成节点的输出，用到一个参数函数$g$，称为局部输出函数。然后可以将$\textbf{h}_v$和$\textbf{o}_v$定义为：
$$
\textbf{h}_v=f(\textbf{x}_v,\textbf{x}_{co[v]},\textbf{h}_{ne[v]}, \textbf{x}_{ne[v]}), \qquad (4.1) \\
\textbf{o}_v=g(\textbf{h}_v,\textbf{x}_v), \qquad (4.2)
$$
其中$\textbf{x}$代表输入特征，$\textbf{h}$代表隐藏状态。

...

...







# Chapter 5.  Graph Convolutional Networks

## 5.1 Spectral Methods

谱方法使用图的谱表达。

### 5.1.1 Spectral Network

卷积操作可以通过计算图的Laplacian矩阵的特征分解在傅里叶域中进行定义。这个操作可以定义为信号（每个节点的标量）$\textbf{x}\in \mathbb{R}^{N}$和一个参数为$\theta\in \mathbb{R}^{N}$的滤波器$\textbf{g}_{\theta}=\text{diag}(\theta)$的乘积：
$$
\textbf{g}_{\theta}\star \textbf{x}=\textbf{Ug}_{\theta}(\Lambda)\textbf{U}^T\textbf{x}, \qquad (5.1)
$$
其中
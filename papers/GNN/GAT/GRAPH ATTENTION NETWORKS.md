# GRAPH ATTENTION NETWORKS

## ABSTRACT
提出一个全新的神经网络架构——图注意力网络(GATs)，它在图结构数据上进行操作，利用掩码自注意力层来解决以前方法（图卷积或近似方法）的缺陷。

## 1 INTRODUCTION
卷积神经网络作用的数据必须是网格状结构。这样的架构通过将带可学习参数的局部滤波器作用于所有位置而达到高效地重复使用。

但是很多任务的数据无法以网格状结构来表示，而是存在于一个不规则的域中。例如3D meshes,社交网络、电信网络、生物网络等。这种数据通常可以用图的形式表示。

blahblahblah

在将卷积泛化到图域的兴趣日益高涨。在这一方面的推进一般分称两类：谱方法和非谱方法。

一方面，谱方法使用图的谱表示并已成功应用于节点分类任务。
在所有已经提出的谱方法中，学习的滤波器依赖于Laplacian特征基，而它则来自于图结构。因此在一个特定结构上训练的模型无法直接应用于另一个不同结构的图。

另一方面，非谱方法直接在图上定义了卷积，在一组稀疏的近邻上进行操作。这一方向的一大挑战是定义一个算子，使其可以在不同尺寸的邻接上进行操作并维护CNNs的共享权重。其中GraphSAGE，介绍了一个计算节点特征的归纳式方法。这个方法对每个结点采样固定尺寸的邻居并在其上使用一个特定的聚合器（如对节点邻居的特征向量求平均，或者将它们传入一个RNN网络得到结果）。这一方法达到了较好的表现。

注意力机制在很多基于序列的任务中几乎已经变成de facto（实际上的）标准。注意力机制的一个优点是可以处理不同尺寸的输入，关注于输入中最相关的部分来得出结论。当使用注意力机制计算一个单独序列的表达，通常称为*自注意力*或者*内部注意力*。

## 2 GAT ARCHITECTURE

### 2.1 GRAPH ATTENTIONAL LAYER
单独的图注意力层的描述

层的输入是一组节点特征$\textbf{h}=\{h_1,h_2,...,h_N\},h_i \in \mathbb{R}^F$，其中$N$是结点的数量，$F$是特征的维度。这一层生成一组新的节点特征（不同维度）$\textbf{h}'=\{h_1',h_2',...,h_N'\},h_i' \in \mathbb{R}^{F'}$作为输出。

为了得到足够的表达能力将输入特征变换成更高级特征，至少需要1个可学习的线性变换。因此在初始化阶段，一个参数为权重矩阵$\textbf{W}\in \mathbb{R}^{F'\times F}$的共享线性变换会应用在每个节点上。然后在节点上使用自注意力——一个共享的注意力机制$a:\mathbb{R}^{F'}\times \mathbb{R}^{F'} \rightarrow \mathbb{R}$来计算注意力相关系数：
$$
e_{ij}=a(\textbf{W}h_i,\textbf{W}h_j) \qquad (1)
$$
代表j节点对于i节点的重要性。对于最一般的公式，模型允许每个节点注意到其余所有节点，得到全部结构信息。通过使用掩码注意力将图结构注入到这个机制中——只计算节点$j\in \mathcal{N}_i$的$e_{ij}$，其中$\mathcal{N}_i$是节点i的一部分邻居。实验中只采用一阶邻居。为了使节点间的相关系数方便比较，在全部j上使用softmax函数进行归一化：
$$
a_{ij}=\text{softmax}(e_{ij})=\frac{\exp(e_{ij})}{\sum_{k\in \mathcal{N}_i}\exp(e_{ik})}. \qquad (2)
$$
在我们的实验中，注意力机制a是一个单层的前馈神经网络，参数为一个权重向量$\vec{\textbf{a}}\in \mathbb{R}^{2F'}$，并使用LeakyReLU($\alpha=0.2$)非线性函数。注意力机制计算的相关系数（如Figure 1左图所示）完全展开的形式可以表示为：
$$
a_{ij}=\frac{\exp\Big(\text{LeakyReLU}\Big(\vec{\textbf{a}}^T[\textbf{W}h_i\lVert \textbf{W}h_j]\Big)\Big)}{\sum_{k\in \mathcal{N}_i}\exp\Big(\text{LeakyReLU}\Big(\vec{\textbf{a}}^T[\textbf{W}h_i\lVert \textbf{W}h_k]\Big)\Big)} \qquad (3)
$$
其中$\cdot^T$代表转置，$\lVert$是拼接操作。

获取归一化的注意力系数之后，用于计算特征的线性组合，作为每个节点的最终特征（还要经过一个激活函数$\sigma$）：
$$
h_{i}'=\sigma \Bigg(\sum_{j\in \mathcal{N}_i}\alpha_{ij}\textbf{W}h_j\Bigg). \qquad (4)
$$
为了稳定自注意力的学习过程，发现将我们的机制扩展到多头注意力会有收益。在公式(4)上采用K个独立地注意力机制，然后将它们的特征拼接到一起，得到输出特征表示：
$$
h_i'=\mathop{\lVert}\limits_{k=1}^K \sigma \Bigg(\sum_{j\in \mathcal{N}_i}\Bigg)
$$
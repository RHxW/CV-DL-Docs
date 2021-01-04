# Learning to Cluster Faces on an Affinity Graph

## 1. Introduction

要利用无标签数据的一个自然的想法就是将他们聚类成“伪类别”，这样就可以把这些数据当作有标签数据使用，传入有监督学习流程中。这些聚类一般采取无监督方式，如K-means，spectral clustering，hierarchical clustering和approximate rank-order，将无标签的人脸分组。这些方法依赖于简单的假设，例如K-means隐式地假设每一个族的样本都围绕一个中心；spectral clustering需要族的大小相对平衡等。所以这些方法缺少处理复杂聚类结构的能力，因此常会得到带噪音的聚类结果，特别是在真实世界大尺寸数据集上使用的时候。这个问题严重地限制了这种方法的性能提升。

因此，要想高效利用无标签的人脸数据，就需要开发一种能够处理现实中经常出现的复杂聚类结构的高效的聚类算法。显然，依赖于简单假设不能提供这样的能力。所以本文中，我们探索了一个从本质上不同的方法，也就是学习如何从数据中进行聚类(cluster from data)。特别地，我们希望利用图卷积网络所具有的强大的表达能力来捕捉人脸聚类中的一般模式并发挥其作用来帮助对无标签数据进行划分。

提出一个基于图卷积网络的人脸聚类框架。其流程与用于实例分割的Mask R-CNN类似，例如：生成proposals，识别正类，然后用mask调整等。这些步骤是通过一个迭代proposal生成器分别完成的，这个生成器基于超顶点(super-vertex)，和一个图检测网络还有一个图分割网络。需要注意的是尽管我们受Mask R-CNN启发，我们的框架仍然与其有着本质上的区别：前者是在2D图片grid上操作，后者在一个特定结构的affinity图上操作。如Figure 1所示，根据图卷积网络学习到的结构模式而非简单假设，我们的框架可以解决复杂结构的聚类。

![Figure 1](1.png"Figure 1")



## 2. Related Work

**Face Clustering** 聚类是机器学习中的一项基本任务。目前大多数聚类方法都是无监督的。人脸聚类提供一种方式来利用大量的无标签数据。这一方向的研究仍停留在早期阶段。如何对大量人脸数据进行聚类仍然是一个待解决的问题。

早期的方法使用人工特征和经典聚类算法，例如Ho等使用梯度和像素密度作为人脸特征；Cui等使用LBP特征；他们都使用了谱聚类。最近的方法则利用学习的特征。尽管使用了深度特征，这些方法也主要关注于设计一个新的相似度度量，且仍然依赖于无监督方法实现聚类。与上述方法不同，我们的方法基于一个检测分割的模式，使用自顶向下的方式学习如何聚类。这就使模型可以处理复杂结构的聚类。

**Graph Convolutional Networks** 图卷积网络(GCNs)扩展了CNNs的能力，用于处理图结构的数据。GCNs已经展示出的优点包括对复杂图模式的强大的建模能力。

本文使用GCN作为基础结构来获取affinity图上的聚类的模式。

## 3. Methodology

大规模人脸聚类中，聚类模式的复杂变化称为性能提升的主要挑战。为了解决这一挑战，我们探索了一种有监督方法，即基于图卷积网络学习聚类模式。特别地，我们将这个问题视为一个在affinity图上的检测和分割的组合问题。

给定一组人脸数据，提取没一张人脸图片的特征，得到一组特征$\mathcal{D}=\{\textbf{f}_i\}_{i=1}^N$，其中$\textbf{f}_i$是一个d维的向量。要构造affinity图，将每个样本视为一个顶点并使用余弦相似度寻找每个样本的K最近邻。将邻居相连接，就得到了整个数据集的一个affinity图$\mathcal{G}=(\mathcal{V,E})$. 或者，affinity图也可表示成一个对称的邻接矩阵$\textbf{A}\in \mathbb{R}^{N\times N}$，其中如果两个顶点连通，则元素$a_{i,j}$是$\textbf{f}_i$和$\textbf{f}_j$的余弦相似度，否则为0. affinity图是一个有数百万顶点的图，我们希望聚类的结果有如下性质：1）不同类别包含不同标签的图片；2）同一类别图片拥有相同的标签。

![Figure 2](2.png"Figure 2")

### 3.1. Framework Overview

如Figure 2所示，我们的聚类框架由三个模块组成，分别为proposal生成器、GCN-D和GCN-S. 第一个模块生成分类的proposals，例如affinity图中的子图有可能是分类。在获得全部分类建议之后，引入两个GCN模块，GCN-D和GCN-S，形成一个二阶段步骤，首先选择高质量的proposals，然后通过移除其中噪音来对选择的proposals进行调整优化。其中GCN-D实现聚类检测。拿一个类别proposal作为输入，GCN-D评估proposal构成一个类别的可能性。然后使用GCN-S完成分割来优化选中的。特别地，给定一个聚类族，GCN-S估计其内部每一个顶点是噪声的概率，然后通过抛弃离群点的方式对聚类族进行剪枝。根据这两个GCN的输出，就可以高效地获得高质量的聚类族。

### 3.2. Cluster Proposals

受物体检测方法中生成区域proposals的启发，首先生成聚类族proposals而不是直接处理整个affinity图。这个策略可以大幅度减少计算量，因为这种方式下只需要评估有限个候选聚类族。聚类族proposal$\mathcal{P}_i$是affinity图$\mathcal{G}$的一个子图。所有proposals组成一个集合$\mathcal{P}=\{\mathcal{P}_i\}_{i=1}^{N_p}$. 聚类族proposals是基于超顶点生成的，所有超顶点组成一个集合$\mathcal{S}=\{\mathcal{S}_i\}_{i=1}^{N_s}$. 本节首先介绍超顶点的生成，然后在其基础上设计一个组合成聚类族proposals的算法。

**Super-Vertex.** 一个超顶点是一个包含少量距离很近的顶点的子图。因此，很自然地就可以用连通的部分来表示超顶点。但是直接从图$\mathcal{G}$得到的连通部分可能会特别大。为了保持各个超顶点间的高连通性，去除affinity值低于阈值$e_{\tau}$的边，并限制超顶点的数量低于一个最大值$s_{max}$. Algorithm 1展示了生成超顶点集合$\mathcal{S}$的细节流程。一般一个有一百万顶点的affinity图能够划分成50k个超顶点，其中每个平均包含20个顶点。

![Algorithm 1](a1.png"Algorithm 1")

**Proposal Generation.** 与预期的聚类族相比，超顶点是一种保守的形式。尽管一个超顶点中的顶点描述同一个人的可能性很高，但是同一个人的样本可能分布在多个超顶点中。受物体检测中使用的多尺度proposals启发，设计了一个用于生成多尺度聚类族proposals的算法。如Algorithm 2所示，在超顶点之上构建一个更高层级的图，将超顶点的中心作为顶点，将这些中心之间的affinities作为边。在更高层级的图上再使用Algorithm 1得到更大尺寸的proposals. 重复这一操作$I$次，就获得了多尺度的proposals

### 3.3. Cluster Detection

设计一个基于图卷积网络(GCN)的模块用于从生成的聚类族proposals中选择高质量聚类族。这里关于质量的衡量标准有两个，分别是IoU和IoP评分。给定一个聚类族proposal$\mathcal{P}$，这两个评分定义为：
$$
IoU(\mathcal{P})=\frac{|\mathcal{P}\cap \widehat{\mathcal{P}}|}{|\mathcal{P}\cup \widehat{\mathcal{P}}|}, \qquad IoP(\mathcal{P})=\frac{|\mathcal{P}\cap \widehat{\mathcal{P}}|}{\mathcal{P}}, \qquad(1)
$$
其中$\widehat{\mathcal{P}}$是所有顶点带标签$l(\mathcal{P})$的gt集合，$l(\mathcal{P})$是聚类族$\mathcal{P}$的多数标签，即$\mathcal{P}$中出现最多的标签。直观来看，IoU反映了$\mathcal{P}$和$\widehat{\mathcal{P}}$的接近程度；IoP反映纯度，即$\mathcal{P}$的顶点中标签占多数的比例。

**Design of GCN-D.** 我们假设高质量聚类族通常表现出节点间特定的结构模式。引入一个GCN来识别这样的聚类族。特别地，给定一个聚类族proposal$\mathcal{P}_i$，GCN将可视化特征和其顶点（表示为$\textbf{F}_0(\mathcal{P}_i)$）和affinity子矩阵（表示为$\textbf{A}(\mathcal{P}_i)$）作为输入，同时预测IoU和IoP分数。

GCN网络由L层组成，每一层的计算可以表示为：
$$
\textbf{F}_{l+1}(\mathcal{P}_i)=\sigma \Big(\tilde{\textbf{D}}(\mathcal{P}_i)^{-1}(\textbf{A}(\mathcal{P}_i)+\textbf{I})\textbf{F}_l(\mathcal{P}_i)\textbf{W}_l\Big), \qquad(2)
$$
其中$\tilde{\textbf{D}}=\sum_j\tilde{\textbf{A}_{ij}(\mathcal{P}_i)}$是一个对角度矩阵。$\textbf{F}_l(\mathcal{P}_i)$包含第l层的embeddings. $\textbf{W}_l$是一个用于转换embeddings的矩阵，$\sigma$是非线性激活函数。直观上，这个公式表述了一个接收每个顶点和其邻接点特征的加权平均作为输入，用$\textbf{W}_l$对它们进行变换，然后把它们传入一个非线性激活函数的过程。这个过程与CNN中的典型块类似，只不过是在拥有不确定拓扑结构的图上进行的操作。在高层embeddings $\textbf{F}_L(\mathcal{P}_i)$上使用一个覆盖$\mathcal{P}_i$内全部顶点的最大池化，得到一个提供总览信息的特征向量。然后使用两个全连接层分别预测IoU和IoP分数。

**Training and Inference.**  给定一个带类别标签的训练集，可以根据公式(1)获取每个聚类族proposal$\mathcal{P}_i$的gt IoU和IoP分数。然后训练GCN-D模块，目标是最小化gt和预测分数间的MSE. 
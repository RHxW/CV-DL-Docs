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

给定一组人脸数据，提取每张人脸图片的特征，得到一组特征$\mathcal{D}=\{\textbf{f}_i\}_{i=1}^N$，其中$\textbf{f}_i$是一个d维的向量。要构造affinity图，将每个样本视为一个顶点并使用余弦相似度寻找每个样本的K最近邻。将邻居相连接，就得到了整个数据集的一个affinity图$\mathcal{G}=(\mathcal{V,E})$. 或者，affinity图也可表示成一个对称的邻接矩阵$\textbf{A}\in \mathbb{R}^{N\times N}$，其中如果两个顶点连通，则元素$a_{i,j}$是$\textbf{f}_i$和$\textbf{f}_j$的余弦相似度，否则为0. affinity图是一个有数百万顶点的图，我们希望聚类的结果有如下性质：1）不同类别包含不同标签的图片；2）同一类别图片拥有相同的标签。

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

**Training and Inference.**  给定一个带类别标签的训练集，可以根据公式(1)获取每个聚类族proposal$\mathcal{P}_i$的gt IoU和IoP分数。然后训练GCN-D模块，目标是最小化gt和预测分数间的MSE. 从实验中可以看到，在没有使用任何高级技术的前提下，GCN可以给出准确的预测。推理阶段，使用训练好的GCN-D来预测每个proposal的IoU和IoP分数。IoU分时会用在3.5节中来保留较高IoU的proposals. IoP分数会用在下一阶段来决定是否保留一个proposal.

### 3.4. Cluster Segmentation

GCN-D识别出的前几个proposals可能并不纯。这些proposals可能仍然包含一些离群点，需要消灭它们。所以开发了一个聚类分割模块，称为GCN-S，用于排除proposal包含的离群点。

**Design of GCN-S.** GCN-S的结构与GCN-D相似。区别主要是预测的值。GCN-S对每个顶点输出一个概率值来预测它是一个真正的成员还是一个离群点。

![Figure 3](3.png"Figure 3")

**Indentifying Outliers.** 要训练GCN-S，需要准备gt. 一种自然的方式是将那些label值与多数label不同的顶点都当作离群点对待。但是，如Figure 3所示，当一个proposal中的不同label顶点数量差不多的时候，这个方法就会遇到问题。为了避免向人工定义的离群点过拟合，激励模型去学习不同的分割模式。只要分割的结果包含一个类别的顶点，无论其是否是多数标签，都将其视为合理的解。特别的，随机选择proposal中的一个顶点作为seed. 将一个值接在顶点的特征之后，当顶点为选中的seed时，值为1，否则为0. 拥有与seed相同label的顶点被视为正顶点，其他的则被视为离群点。多次使用这个方法，随机选择seed，就得到了每个proposal$\mathcal{P}$的多个训练样本。

**Training and Inference.** 使用上述流程，可以从保留的proposals中准备一个训练样本集。每个样本包含一组特征向量，每一个特征向量对应一个顶点，和一个邻接矩阵，同样是一个二元向量，宝石顶点是否为正类。然后使用顶点的二元交叉熵作为损失函数来训练GCN-S模块。推理过程中，同样对一个生成的聚类族proposal提出对各假设，并只保存最多正顶点的预测结果（阈值为0.5）。这一策略避免了被选一个与少量正顶点相关联的顶点当作seed的情况误导。

只将IoP介于0.3~0.7之间的proposals输入GCN-S中。因为当proposal很纯净时，离群点通常是困难样本，需要保留。当proposal很不纯净时，有可能其中任一类别都不占主要地位，因此这个proposal也许不适合GCN-S来处理。根据GCN-S的预测结果，将离群点从proposals中移除。

### 3.5. De-Overlapping

经过上述三个阶段可以得到一组聚类族。但是不同的聚类族之间仍有可能存在重叠的现象，即共享某些顶点。这可能引起在其上进行的人脸识别训练产生反向效果。因此提出一个简单快捷的去重叠算法来解决这个问题。首先对聚类族proposals按IoU得分降序排序。按顺序取proposal并去除其中在之前出现过的顶点。算法描述见Algorithm 3所示。

与物体检测中使用的非极大值抑制（NMS）相比，去重叠方法更有效。特别地，前者的复杂度为$O(N^2)$，后者的复杂度为$O(N)$.可以通过设置IoU阈值的方式进一步提升去重叠速度。

![Algorithm 3](a3.png"Algorithms 3")

## 4. Experiments

### 4.1. Experimental Settings

blahblahblah

### 4.2. Method Comparison

#### 4.2.1 Face Clustering

将提出的方法与一系列聚类基准进行了对比。这些方法如下：

**(1) K-means**，最常用的聚类算法。给定聚类数量$k$，K-menas最小化类内方差

**(2) DBSCAN**，一种基于密度的聚类算法。根据一个设计好的密度准则提取聚类族，剩下的稀疏背景则视作噪声。

**(3) HAC**，层次聚类是一种自底向上的方法，根据某些准则迭代地合并距离近的聚类族。

**(4) Approximate Rank Order**，HAC的一种形式。只迭代一次

**(5) CDP**，提出一种基于图的聚类方法。在自底向上的方法中更好地利用了成对的关系。

**(6) GCN-D**，本文提出的第一个方法。使用一个GCN来有监督地学习聚类模式。

**(7) GCN-D + GCN-S**，本文提出方法的二阶段版本。引入GCN-S来优化GCN-D的输出，它检测并排除噪音。

**Results** 为了控制实验的时间，随机选择了一部分数据用于验证，其中包含8573个id的580k张图片。Table 1对比了不同方法在这个数据集上的性能表现。对它们性能的评估采用的标准是F-score和时间消耗。同时也展示了聚类的数量，成对精度和成对召回，用于更好地理解各方法的优缺点。

![Table 1](t1.png"Table 1")

结果表明：（1）K-means的性能受聚类个数$k$影响严重。测试了一定范围内的多个$k$值，取F-score最高的结果。（2）DBSCAN精度高但是召回低。在面对大量人脸聚类的大密度差异情况下可能会失败。（3）HAC的结果比前两个方法更鲁棒。注意到标准的层次聚类算法需要消耗$O(N^2)$内存，当$N$达到580k这么大的时候会超内存。所以使用了一个自适应层次聚类算法，只需要$O(Nd)$的内存。（4）Approximate Rank Order由于它只迭代一次的特点使它效率很高，但是其性能表现较差。（5）CDP，利用无标签数据用于人脸识别的方法，在精准度和召回率之间达到了一个较好的平衡。为了公平地比较，我们对比了单模型版的CDP. 注意到CDP和本文在思想上使互补的，可以将二者结合来进一步提升性能。（6）我们的方法使用GCN来学习聚类模式。同时提升了精准度和召回率。Table 2展示了我们的方法具有鲁棒性且可以应用于有不同分布的数据集上。由于GCN使用多尺度聚类proposals训练，它可以更好地捕捉目标聚类族的性质。如Figure 8中所示，我们的方法能够精确定位一些有复杂结构的聚类族。（7）GCN-S模型进一步优化第一阶段的聚类族proposals结果。通过轻微牺牲召回的方式提升精度，得到总体的性能提升。

![Table 2](t2.png"Table 2")

*****

![Figure 8](8.png"Figure 8")



#### 4.2.2 Face Recognition

在无标签数据上使用训练好的聚类模型获取伪标签。调查了无标签数据使用伪标签如何增强人脸识别性能。步骤如下：

（1）使用有标签数据以有监督的形式训练初始识别模型；

（2）在标签集合上使用初始模型得到的特征表达来训练聚类模型；

（3）在不同数量（1，3，5，7，9部分）的分组的无标签数据上使用聚类模型，将数据与伪标签关联；

（4）使用整个数据集训练最终的识别模型，其中包含原始有标签数据和伪标签关联数据。

其中，将只在1部分上训练的模型作为下界，使用全部gt标签训练的模型当作上界。所有聚类方法都会给一张无标签图片分配一个聚类族。将这个聚类族id作为该图片的伪标签。

![Figure 5](5.png"Figure 5")

Figure 5展示了人脸聚类的性能对人脸识别的影响很大。对于K-means和HAC，尽管召回率不错，但是低精度表明其预测的是有噪音的聚类。当无标签和有标签数据的比例较小的时候，有噪音的聚类会严重削弱人脸识别的训练效果。随着比例的提升，无标签数据会平衡这一噪声影响。但是整体的提升有限。CDP和本文提出的方法都受益于无标签数据的增长。



### 4.3. Ablation Study

随机选择了一部分无标签数据，其中包含8573个id的580k张图片，来研究本框架的一些重要设计选择。

#### 4.3.1 Proposal Strategies

聚类族proposals的生成是本框架的基础模块。当固定$K=80$而使用不同的$I,e_{\tau},s_{max}$时，生成不同尺度的大量proposals. 一般来说，大量的proposals会得到更好的聚类效果。这里在性能和计算量上有一个取舍，决定于合适的proposals数量的选择。如Figure 4所示，每个点都代表某个数量proposals的F-score. 不同的颜色表示不同的迭代步数。（1）当$I=1$时，只会使用Algorithm 1生成的超顶点。通过选择不同的$e_{\tau}$，使用更多的proposals来提升F-score. 当数量达到100k时性能逐渐饱和。（2）当$I=2$时，会将不同超顶点的组合添加到proposals中。平衡了超顶点间的相似度，因此有效地增大了proposals的感受野。只加入少量的proposals，F-score提升了5%.（3）$I=3$时，进一步合并前一阶段的相似的proposals来生成更大尺度的proposals，这样做可以持续提升proposal尺度，同时会引入更多的噪音，因此性能提升达到饱和。

![Table 3](t3.png"Table 3")

#### 4.3.2 Design choice of GCN-D

尽管GCNs的训练不需要高级技术，但仍然有一些重要的设计选项。如Table 3中a,b,c所示，池化方式对F-score的影响很大。与最大池化相比，平均池化与和池化会降低性能。对于和池化，它对顶点数量敏感，倾向于生成大的proposals. 而大的proposals会得到高的召回率（80.55）和低的准确率（40.33），导致F-score低。而平均池化更好地描述了图的结构，但是可能受proposal中离群点的影响。除了池化方法外，Table 3中的c和d展示了没有定点特征会显著降低GCN的预测准确率。这一点揭示了在GCN的训练中平衡顶点特征和图结构时很有必要的。另外，如Table 3中c,e和f所示，增加GCN的通道数可以提升其表达能力，但是更深的网络可能使顶点的隐层特征趋于相似，结果与平均池化相似。

#### 4.3.3 GCN-S

本框架中，使用GCN-S作为GCN-D之后的去噪音模块。但是它可以作为一个独立的模块与之前的方法相结合。给定K-means，HAC和CDP的结果，将它们当作聚类族proposals并传入GCN-S中。如Figure 6所示，GCN-S能够通过抛弃内部离群点来提升它们的聚类性能，对于不同的方法，提升大概在2%-5%之间。

![Figure 6](6.png"Figure 6")

#### 4.3.4 Post-process strategies

NMS是一种物体检测中广泛使用的后处理技术，可以作为去重叠的可选方法。根据不同的IoU阈值，保留最高IoU的proposal同时抑制其他重合的proposals. 其计算复杂度为$O(N^2)$. 与NMS相比，去重叠不需要抑制其他proposals因此保留了更多的样本，提成了聚类的召回率。如Figure 7所示，取重叠获得了更好的聚类性能，而且可以在线性时间内计算完成。
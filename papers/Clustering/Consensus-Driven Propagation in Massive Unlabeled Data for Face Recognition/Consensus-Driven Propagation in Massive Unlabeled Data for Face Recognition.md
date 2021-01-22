# Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition

## 1 Introduction

问题：人脸数据集人工标注时间成本太高，几乎所有数据集都有噪音问题

为了解决上述问题，我们将视角从获取更多人工标注的标签转向利用更多无标签的数据上。和大规模id标注不同，无标签的图片很容易获取。比如，用一个人脸检测器加爬虫软件就可以轻易获取大量的非限制场景人脸图片或视频。现在的主要问题变成了如何利用已有的大量无标签数据来提高大规模人脸识别效果。

我们的目标是最大化利用无标签数据，使最终的性能表现与使用标注数据的性能表现相近。这里一个关键结论是尽管无标签数据无法提供直接的语义类别，但是它们可以用图来表示的内部结构，实际上反映了人脸表示的高维分布。

众所周知，由单独一个模型提出的特征通常容易受到偏置的影响，且对噪声敏感。为解决这个问题，使用一个bottom-up方法，通过首先可靠地识别正样本对的方式来构建图。称为CDP（共识驱动传播，Consensus-Driven Propagation），由两个模块组成：一个“**委员会**(committee)”，用于提供样本对的多角度信息；和一个“**中介**(mediator)”，将全部信息聚合成最终决策结果。

“**委员会**”模块受QBC(query-by-committee)启发，QBC原本用于激活学习。和QBC用于衡量差异不同，我们从由一个基础模型和多个辅助模型的委员会收集consents. committee的异构特性在无标签数据的结构上表现出多个不同视角。将委员会中多数成员赞同的样本对选作正样本对，而不是根据基础模型的置信度选择。因此委员会模块能够从无标签图像中选择有意义的和困难的正样本对，而不是只选出简单样本。除了投票机制之外，我们使用一种新的更有效的“**中介**”来聚合委员会的意见。中介是一个二分类器，其生成是否选择一个样本对的最终决策。我们仔细设计了中介的输入，使之能够覆盖内部结构的分布信息。其输入包括：1）委员会的投票结果，2）样本对之间的相似度，3）样本对之间的局部密度。最后两个输入是通过委员会的全部成员和基础模型衡量的。使用“委员会”和“中介”模块，我们对无标签数据构建了一个鲁棒的共识驱动的图。最终在图上传播伪标签，形成一个辅助任务用来使用无标签数据训练基础模型。



## 2 Related Work

**Semi-supervised Face Recognition.** 在已有少量标注数据的前提下，为了利用大量无标签数据而提出了半监督学习。一般来说，其目的是通过多种方式根据受限的标注数据将标签传播到整个数据集，采用的方法有自训练、协同训练、多视角学习、期望最大化和基于图的方法。

**Query-by-Committee.** QBC是一个依赖于多判别模型来获取不同意见的策略，也因此挖掘机器学习任务的有意义的样本。



## 3 Methodology

首先提出一个总览，本方法主要有三部分组成：

1）**Supervised initialization** - 给定一小部分标注数据，使用全监督的方式独立地训练基础模型和委员会成员。具体来说，基础模型$B$和全部$N$个委员会成员$\{C_i|i=1,2,...,N\}$使用标注好的数据$D_l$学习一个从图像空间到特征空间的映射$\mathcal{Z}$. 对于基础模型，这一过程可表示为：$\mathcal{F}:D_l \mapsto \mathcal{Z}$, 对委员会成员为：$\mathcal{F}_{C_i}:D_l \mapsto \mathcal{Z}, i=1,2,...,N$.

2）**Consensus-driven propagation** - 在无标签数据上使用CDP来选择有价值的样本，并在此基础上推测其标签。其架构见Fig 1. 我们使用第一阶段训练好的模型来提取无标签数据的深度特征，并创建k-NN图来选择有意义的样本对。使用选中的无标签样本对来构建一个共识驱动图，并使用我们的标签传播算法将伪标签分配个各结点。

3）**Joint training using labeled and unlabeled data** - 最终，在多任务学习框架下，使用有标签数据和伪标签数据重新训练基础模型。

![Figure 1](1.png"Figure 1")

> 从图中可以看到，pairs信息只由base feature生成（实际上pairs信息和feature无关）。然后使用committee members的features + pairs信息生成用于Mediator的数据（relationship，affinity，structure和训练用的pairs_label）。先训练mediator，然后就可以用mediator判断每一个pair所连接的两个顶点是否为同一类，就替代了vote策略。

### 3.1 Consensus-Driven Propagation

本节介绍CDP的细节步骤。

**i. Building k-NN Graphs.** 

将无标签数据$D_u$作为输入传入基础模型和委员会成员，提取到深度特征$\mathcal{F}_B(D_u)$和$\mathcal{F}_{C_i}(D_u)$. 使用特征找到$D_u$中每个样本余弦相似度的k最近邻。这样得到了不同版本的k-NN图，$\mathcal{G}_B$对应基础模型，$\mathcal{G}_{C_i}$对应每个委员会成员，一共$N+1$个图。图中的节点代表无标签数据的样本。每一条边定义了一对，基础模型对应图$\mathcal{G}_B$的所有的对构成了后续选择操作的候选者，如Fig 1所示。

**ii. Collecting Opinions from Committee.** 

委员会成员通过使用映射函数$\{\mathcal{F}_{C_i}|i=1,2,...,N\}$将无标签数据映射到特征空间。假设由基础模型创建的图中有任意两个连接的节点$n_0$和$n_1$，它们由不同版本的深度特征$\{\mathcal{F}_{C_i}(n_0)|i=1,2,...,N\}$和$\{\mathcal{F}_{C_i}(n_1)|i=1,2,...,N\}$表示。委员会提供了下述因素：

1）*关系*，两个节点之间的关系$R$. 直观上可以理解为两个节点再每一个委员会成员的视角下是否相邻。
$$
R_{C_i}^{(n_0,n_1)}=
\begin{cases}
1 \quad \mathrm{if} (n_0, n_1)\in \varepsilon(\mathcal{G}_{C_i}) \\
0 \quad \mathrm{otherwise.}
\end{cases}, \quad i=1,2,...,N, \qquad (1)
$$
其中$\mathcal{G}_{C_i}$是第$i$个委员会模型的k-NN图，$\varepsilon$代表图的全部边。

2）*亲和度*，两个节点间的亲和度$A$. 可以通过计算特征空间的相似度得到，映射函数由委员会成员定义。假设我们使用余弦相似度，
$$
A_{C_i}^{(n_0,n_1)}=\cos(\langle \mathcal{F}_{C_i}(n_0), \mathcal{F}_{C_i}(n_1) \rangle), \quad i=1,2,...,N. \qquad (2)
$$
3）每个节点的*局部结构*。这个概念可以参考一个节点的一阶、二阶甚至更高阶邻居的分布。其中一阶邻居扮演了表示一个节点“局部结构”中最重要的角色。这样的分布可以近似为节点$x$和其所有相邻节点$x_k$的相似度分布，其中$k=1,2,...,K$.
$$
D_{C_i}^x=\{\cos(\langle \mathcal{F}_{C_i},\mathcal{F}_{C_i} \rangle), k=1,2,...,K\}, \quad i=1,2,...,N. \qquad (3)
$$
![Figure 2](2.png"Figure 2")

> Fig. 2: **Committee and Mediator.** 这张图片展示了committee和mediator的工作机制。上图中有一些由base model和committee生成的不同的图中采样的节点。每一行中，两个红色的节点是候选对。第一行的节点对被mediator分类为正类，第二行的节点对则被分为负类。Committee给出包括"relation", "affinity", "local structure"在内的多种意见。"local structure"表示为第一级（红色边）和第二级（橙色边）邻居的分布。主要图中只展示了以两个节点中的一个为中心的"local structure"

如Fig 2所示，从base model图中选取一对节点，committee成员给出包括"relation", "affinity", "local structure"在内的多种意见，由于它们天然具有异构性。我们希望通过下一步的mediator从这些不同的意见中得到一个结论。

**iii. Aggregate Opinions via Mediator.** Mediator的作用是聚合并传送committee成员关于节点对选择的意见。将mediator表示为一个多层感知机(MLP)分类器，虽然其它类型的分类器也是可以的。从base model图提取的所有节点对组成了候选人。mediator会对committee成员的意见重新加权并得出一个最终结果，给每一对节点一个概率值，表示如果一对共有同一个id，则为正，有不同id则为负。

每个pair$(n_0,n_1)$输入到mediator的是一个拼接起来的向量，其由三部分组成（此处为简单起见，将$B$表示为$C_0$）：

1) "relationship vector" $I_R\in\mathbb{R}^N: I_R=\Big(...R_{C_i}^{(n_0,n_1)}\Big),i=1,2,...,N$, 来自committee

2)  "affinity vector" $I_A\in \mathbb{R}^{N+1}:I_A=\Big(...A_{C_i}^{(n_0,n_1)}...\Big), i=0,1,2,...,N$, 来自于base model和committee

3) "neighbors distribution vector" 包括"mean vector"$I_{D_{mean}} \in \mathbb{R}^{2(N+1)}$和"variance vector"$I_{D_{var}}\in\mathbb{R}^{2(N+1)}$:
$$
I_{D_{mean}}=\Big(...E(D_{C_i}^{n_0})...,\quad...E(D_{C_i}^{n_1})...\Big),i=0,1,2,...,N,\\
I_{D_{var}}=\Big(...\sigma(D_{C_i}^{n_0})...,\quad...\sigma(D_{C_i}^{n_1})...\Big),i=0,1,2,...,N, \qquad(4)
$$
对于每个节点，来自于base model和committee. 然后得到一个维度为$6N+5$的结果。mediator使用$D_l$训练，目标是最小化对应的交叉熵损失函数。测试的时候将$D_u$中的数据对传入mediator中，收集那些正概率较高的样本。由于大多数正类样本对是冗余的，所以设置了一个较高的阈值用于选择样本对，因此牺牲召回率来获取较高的正样本对准确率。

**iv. Pseudo Label Propagation.** 前一步中由mediator选出的样本对组成了一个”共识驱动图(CDG, Consensus-Driven Graph)“，其中的边的权重代表其连接的节点对为正的概率。注意这张图不一定是连通图。与传统的标签传播算法不同，我们不在图上假定有标签的节点。为接下来的模型训练做准备，我们对基于节点间的连通性的伪标签进行传播。为了传播伪标签，发明了一个用来识别连接部分的简单高效的算法。首先，在图中根据当前边找到连通的部分并将其加入到一个队列中。对每个识别过的部分，如果其包含的节点数量大于一个预先设定的值，则去掉其内部分数低的边，找到与其连通的其他部分，将新的分离的部分加入到队列中。如果一部分的节点数量少于预设值，则将该部分的所有节点标记为一个新的伪标签。迭代这一过程直到队列为空，也就是所有符合条件的部件都标注完毕。

### 3.2 Joint Training using Labeled and Unlabeled Data

一旦给无标签数据分配了伪标签，就可以使用这些数据增广有标签数据并更新base model. 由于不知道两个数据集的id交集，将学习作为一个多任务训练方式，如Fig 3所示。两个任务的CNN架构与base model一模一样，且共享权重。这两个CNN后都接了一个全连接层，将特征映射到对应的标签空间。整体的优化目标是$\mathcal{L}=\lambda \sum_{x_l,y_l}\ell(x_l,y_l)+(1-\lambda)\sum_{x_u,y_a}\ell(x_u,y_a)$,其中loss，$\ell(\cdot)$，与训练base model和committee所用的相同。在接下来的实验中，使用softmax作为损失函数。但是对于CDP而言，其loss并无特别的限制。在上面这个公式中，$\{x_l,y_l\}$代表有标签数据，$\{x_u,y_a\}$代表无标签数据和其被分配的标签。$\lambda\in(0,1)$是用于平衡两部分的权重。其值根据有标签图片和无标签图片的比例确定。从头开始训练模型。

![Figure 3](3.png"Figure 3")



## 4 Experiments

**Training Set.** blahblahblah

**Testing Sets.** blahblahblah

**Committee Setup.** 为了创造一个高度异构的committee，使用了包括ResNet18, ResNet34, ResNet50, ResNet101, DenseNet121, VGG16, Inception V3, Inception-ResNet V2和NASNet-A的一个较小变体在内的CNN架构。实验中所使用的committee的成员数量为8，我们同样实验了0-8的数量。使用有标注数据训练所有这些网络，结果如Table 1所示。

![Table 1](t1.png"Table 1")

**Implementation Details.** mediator是一个有两个隐藏层的MLP分类器，每一个隐藏层包含50个节点，使用ReLU作为激活函数。测试时，将概率阈值设置为0.96，用于选择高置信度的样本对。

### 4.1 Comparisons and Results

**Competing Methods.** 1）有监督深度特征提取器+层次聚类（Hierarchical clustering）：我们准备了使用有监督深度特征提取器的层次聚类的一个强大的基准。与其他聚类方法相比，层次聚类是处理大量数据时的很实用的方法。集群被分配了伪标签，并用于增广训练集。为了获得最佳性能，我们使用验证集且忽略只有一张图片的集群，仔细调整层次聚类的阈值。2）利用原始committee投票来选择样本对：如果全部committee成员都对某个样本对投票，则选择它。

**Benchmarking.** 如Fig 4所示，CDP方法表现很好。从结果中可以看到：

1）与没有无标签数据的下限相比（无标签：有标签的比例时0：1），CDP在不同数量未标注数据的情况下获得了巨大且稳定的提升。

2）CDP大幅度超越了层次聚类基准，齐平甚至高于全监督方法。

![Figure 4](4.png"Figure 4")

3）采用mediator方案的CDP的性能比采用原始投票法要好，这表示mediator聚合committee意见的能力更强。

4）在IJB-A人脸验证任务中，这两种CDP的设置都超越了全监督方法。

**Visual Results.** 在Figure 9中可视化了CDP的结果。可以观察到CDP在识别label任务上的准确性较高而不受不同的背景、表情、姿态和光照的影响。同样可以看到CDP在选择配对候选时是有选择性的，它自动忽略了：1）错误标注的不属于任何id的人脸；2）很低质量的样本，包括高度模糊和卡通图片。这解释了为什么CDP可以超越全监督基准。

![Figure 9](9.png"Figure 9")



### 4.2 Ablation Study

在验证集上进行了消融实验，结果如Table 2所示。选中样本对的召回和准确率更高，得到的共识驱动图就更好，因此可以提升分配标签的质量。对于分配好的标签，pairwise召回和精准率反映了标签的质量，并与最终性能直接相关。更高的pairwise召回率表明一个类别中的真实样本更多，这对后续训练很重要。更高的pairwise精准率表明类中噪音更小。

**The Effectiveness of "Committee".** 当改变committee成员数量的时候，为了方便起见，调整样本对的相似度阈值来获得固定的召回率。随着committee数量的增加，一个有趣的现象是精度的峰值出现在数量为4的时候。但是此时分配标签的质量并不是最好的，而质量最好则发生在数量为6-8的时候。这一现象表明了更多的committee成员会带来更有意义的样本而不仅仅是正确的样本对。这一结论与我们的假设相匹配，即committee可以选择更多的与基础模型相关的困难正样本对。

**The Effectiveness of "Mediator".** 对于mediator，研究了不同输入设置的影响。当只有关系向量$I_R$作为输入时，指示器的值接近于直接投票。然后亲和向量$I_A$大幅度提升选中样本对的召回和精准率，同样提升了分配标签的pairwise的召回和精准率。临近分布向量$I_{D_{mean}}$和$I_{D_{var}}$进一步提升了分配标签的质量。效果的提升源于信息的这些方面，因此mediator的性能要比原始投票要好。

### 4.3 Further Analysis

**Different Base Architectures.** 

![Table 3](t3.png"Table 3")

**Different $k$ in k-NN.** 将正样本对的概率阈值固定为0.96. 如Table 4所示，更大的$k$会得到更多的选中样本对，因此得到一个更密集的共识驱动图，但是其准确率几乎不变。

**Committee Heterogeneity.** 为了研究committee异构性的影响，使用同类架构committee进行试验。blahblahblah. 结论：异构有用！

**Inside Mediator.** 结论：亲和度向量和均值矩阵影响更大！

**Incorporating Advanced Loss Functions.** 见Table 6

![Table 6](t6.png"Table 6")

**Efficiency and Scalability.** CDP的运行时间呈线性增长，因此即高效又可扩展。
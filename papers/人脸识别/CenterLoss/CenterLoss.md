# A Discriminative Feature Learning Approach for Deep Face Recognition

为了提高深度学习特征的分辨能力，提出人脸识别任务的一个新监督信号，CenterLoss，它同时对每个类别的深度特征学习一个中心并对特征和对应类别中心间的距离进行惩罚。在softmax loss和center loss的联合监督下，可以训练一个鲁棒的CNN用来获取深度特征，其具有两个关键学习目标，尽可能的类间分散和类内紧凑。



## 1 Introduction

最常使用的CNN可以实现特征学习和标签的预测，其将输入数据映射到深度特征（最后隐藏层的输出），然后输出预测的标签，如Figure 1所示。

![Figure 1](1.png"Figure 1")

对于一般的物体、场景或动作识别任务，测试样本中可能出现的类别都包含在训练集中了，也称为封闭集合识别(close-set identification)。因此，预测的标签控制其性能而softmax loss可以直接解决分类问题。这样，标签的预测就和一个线性分类器差不多，学到的特征也倾向于可分离。

对于人脸识别任务，深度学习特征不仅需要是可分离的，还需要是具有分辨性的。因为预先收集好训练中所有可能的测试id是不现实的，所以CNN中的标签预测并不总是可应用的。深度学习的特征需要具有足够强的分辨性和泛化性来识别新的、未见过且没有标签的类别。特征的分辨能力同时体现在紧凑的类内偏差和分离的类间差异上，见Figure 1. 对于具有分辨性的特征可以使用NN或kNN算法分类，而不需要依赖标签的预测。但是softmax loss只能促进特征的可分离性。得到的特征对于人脸识别来说有效性不够。

为CNN有分辨性的特征学习构建高效损失函数是很有意义的。因为SGD基于mini-batch优化CNN，它无法很好地反映深度特征的全局分布。由于训练集数据量巨大，将所有训练样本放入每个循环中是不现实的。替代思路包括对抗loss和triplet loss，分别对图像对和三元组构建loss函数。但是与图像样本相比，训练图像对或者三元组的数量增长地非常夸张。这样必然导致收敛缓慢和不稳定。当然可以通过细心挑选图像对或三元组来部分缓解这一问题。但是这样做会极大增加计算复杂度，而且训练过程会变困难。

本文提出一个新的损失函数称为center loss，目的是高效地增强神经网络深度特征的分辨能力。对每个类别的深度特征学习一个中心（一个与特征向量同维度的向量）。在训练的时候，同步更新中心并最小化深度特征和对应类中心的距离。使用softmax loss 和center loss的联合监督来训练CNN，使用一个超参数来平衡两个监督信号。直观理解，softmax loss强制不同类别间保持距离。center loss将同一类别的特征向其中心靠近。



## 2 Related Work

blahblahblah



## 3 The Proposed Approach

### 3.1 A Toy Example

blahblahblah

### 3.2 The Center Loss

那么如何开发一个高效的loss函数来增强特征的分辨能力呢？直观地讲，关键是在最小化类内偏差的同时保持不同类别特征分离。为了实现这一目的，提出center loss函数：
$$
\mathcal{L}_C=\frac{1}{2}\sum\limits_{i=1}^m \Vert x_i - c_{y_i} \Vert_2^2 \qquad(2)
$$
$c_{y_i} \in \mathbb{R}^d$ 代表深度特征的第$y_i$个类别中心。这个公式有效刻画了类内差异。理想情况下$c_{y_i}$会在特征变化时更新。也就是说，我们得将整个训练集考虑在内并在每次循环中都对每个类别的特征求平均，这么做的效率很低，甚至不可行。因此不能直接使用center loss. 也许这就是这样的center loss到目前为止都没用在CNN中的原因吧。

为了解决这一问题，我们做了两个必要的改进。首先，在mini-batch的基础上更新center而非整个训练集。每个循环中通过计算对应类别特征的平均来得到center（本例中有些center也许不会被更新）. 其次，为了避免少量错误标注的样本引起的大扰动，使用一个标量$\alpha$来控制center的学习率。

$\mathcal{L}_C$关于$x_i$的梯度和$c_{y_i}$的更新方程为：
$$
\frac{\partial \mathcal{L}_C}{\partial x_i}=x_i-c_{y_i} \qquad(3) \\
\Delta c_j=\frac{\sum_{i=1}^m\delta(y_i=j)\cdot (c_j-x_i)}{1+\sum_{i=1}^m\delta(y_i=j)} \qquad(4)
$$
其中，当条件满足时$\delta(condition)=1$，否则为0。$\alpha$限制在$[0,1]$之间。使用softmax loss 和center loss联合监督训练CNN进行分辨性特征学习。公式为：
$$
\mathcal{L}=\mathcal{L}_S + \lambda\mathcal{L}_C \\
=-\sum\limits_{i=1}^m \log{\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^ne^{W_{j}^Tx_i+b_{j}}}}+\frac{\lambda}{2}\sum\limits_{i=1}^m \Vert x_i-c_{y_i}\Vert_2^2 \qquad (5)
$$
显然，center loss监督的CNN可以训练并通过标准SGD进行优化。使用一个标量$\lambda$平衡两个损失函数。传统的softmax loss可被视为这个联合监督的特例，当$\lambda$为0时。在Algorithm 1中总结了联合监督下CNN的学习细节

![Algorithm 1](a1.png"Algorithm 1")

做实验验证了$\lambda$取值对分布的影响

![Figure 3](3.png"Figure 3")



### 3.3 Discussion

**- The necessity of joint supervision. **如果只使用softmax loss作为监督，得到的特征会包含很大的类内差异。另一方面，如果只使用center loss监督CNN，学到的特征和中心会衰减至0（在这里，center loss会非常小）。如果只用其中一个就无法获得分辨能力强的特征。所以将二者合并起来对CNN实行联合监督就很有必要，我们的实验也验证了这一点。

**- Compared to contrastive loss and triplet loss. **blahblahblah



## 4. Experiments

### 4.1 Implementation Details

**Preprocessing. **
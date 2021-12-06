# CurricularFace
## Abstract
本作提出一个全新的适应性课程学习损失CurricularFace，将课程学习的思想运用于loss函数中，得到一个人脸识别的新训练策略，主要是在训练前期针对简单样本，在训练后期针对困难样本。而且提出的CurricularFace会在不同训练阶段适应性地调整简单样本和困难样本的相对重要性。在每个阶段，不同的样本会根据其相应的难度分配不同的重要性。

## 1. Introduction
在人脸识别的训练中使用margin-based的loss可以增强特征的判别能力，但是这种loss函数并不能根据样本的重要性做出有针对性的增强。
对于提升最终的精度，困难样本挖掘是一个很重要的步骤。在通常使用的方法中，OHEM关注一个mini-batch中loss较大的样本，这种方法通过经验的方式决定困难样本所占比例并且完全无视简单样本。Focal loss是一种mining-based的loss函数，使用两个超参数来控制简单样本和困难样本的权重。
本文提出一个新的适应性课程学习loss，称为CurricularFace. 通过适应性的方式将课程学习(CL, Curriculum Learning)应用到人脸识别中。该方法与传统的CL方法主要有两点不同：
1. 课程的构建是适应性的。在传统CL中，样本是按对应难度进行排列的，通常是根据某种预先定义好的先验来确定难度，然后使用这个固定的样本集合来构建课程。而CurricularFace则会在每个mini-batch中随机选择样本，然后通过在线挖掘困难样本的方式，适应性地构建课程。
2. 简单样本和困难样本间的相对重要性差异可以进行动态调整以适应不同的训练阶段。另外，每个困难样本在当前mini-batch中的重要性依赖于其自身的困难程度。

实际上，mini-batch中被误分类的样本会选作困难样本，并通过调整样本与非gt类中心向量间cosine距离的调制系数$I(t,\cos \theta_j)$来调整其对应的权重。为了在整个训练过程中达到适应性课程学习的目的，设计了一个新的系数函数$I(\cdot)$，它受两个因素影响：
1. 适应性确定的参数t，利用样本与对应gt类中心的动态余弦相似度，从而避免手动调整。
2. 角度$\theta_j$定义了困难样本的的难度

## 2. Related Work
**Curriculum Learning.** 先学习简单样本，再学习困难样本是CL中的一种常见策略。CL的关键问题是如何定义每个样本的困难程度。以前都是针对不同问题设计不同的方法，所以不能通用。针对这一问题提出了SPL(Self-Paced Learning)，将loss较低的样本视为简单样本并在训练过程中进行加强。CurricularFace和SPL的差别在于：
1. CF在训练的初始阶段更关注简单样本，在后续阶段则会加强困难样本
2. 提出一个新的函数$N(\cdot)$用来计算负的余弦相似度

## 3. The Proposed CurricularFace
### 3.1. Preliminary Knowledge on Loss Function
原始softmax loss的形式为：
$$
\mathcal{L}=-\log \frac{e^{W_{y_i}x_i+b_{y_i}}}{\sum_{j=1}^n e^{W_{j}x_i+b_{j}}}
$$
后续针对人脸识别任务设计的基于softmax loss的各种变体形式可以写成：
$$
\mathcal{L}=-G(p(x_i))\log \frac{e^{sT(\cos\theta_{y_i})}}{e^{sT(\cos\theta_{y_i})}+\sum_{j=1,j\ne y_i}^n e^{sN(t,\cos\theta_{j})}}, \\.\\
\text{where} \quad p(x_i)=\frac{e^{sT(\cos\theta_{y_i})}}{e^{sT(\cos\theta_{y_i})}+\sum_{j=1,j\ne y_i}^n e^{sN(t,\cos\theta_{j})}}
$$
$p(x_i)$是预测的gt概率，$G(p(x_i))$是一个指示函数。$T(\cos \theta_{y_i})$和$N(t,\cos \theta_{j})=I(t,\cos \theta_{j})\cos \theta_{j}+c$是调节正例和负例余弦相似度的函数，c是常数项，$I(t,\cos \theta_{j})$代表负例余弦相似度的调节系数。对于margin-based的loss函数，例如ArcFace，$G(p(x_i))=1$,$T(\cos \theta_{y_i})=\cos(\theta_{y_i}+m)$,$N(t,\cos \theta_{j})=\cos \theta_j$，它只修改了每个样本的正例的余弦距离来增强特征判别能力。如Figure 1所示，每个样本的负例余弦相似度的$I(\cdot)$固定为1. 后来的工作，MV-Arc-Softmax通过增大困难样本的$I(t,\cos \theta_{j})$来增强困难样本权重。也就是$G(p(x_i))=1$,$N(t,\cos \theta_{j})$形式如下：
$$
N(t,\cos \theta_{j})=
\begin{cases}
    \cos \theta_j, \qquad\qquad\qquad T(\cos \theta_{y_i})-\cos \theta_j \ge 0\\
    t\cos \theta_j+t-1, \qquad T(\cos \theta_{y_i})-\cos \theta_j < 0
\end{cases}
$$
判断为简单的样本，它的负例余弦相似度不变；如果是困难样本，它的负例余弦相似度会变成$t\cos \theta_j+t-1$.

### 3.2. Adaptive Curricular Learning Loss
提出的loss函数也在上述的通用框架内，其$G(p(x_i))=1$，正负例的余弦相似度函数如下：
$$
T(\cos \theta_{y_i})=\cos(\theta_{y_i}+m), \\
N(t,\cos \theta_{j})=
\begin{cases}
    \cos \theta_j, \qquad\qquad\qquad T(\cos \theta_{y_i})-\cos \theta_j > 0\\
    \cos \theta_j(t+\cos \theta_j), \qquad T(\cos \theta_{y_i})-\cos \theta_j < 0
\end{cases}
$$
其实正例余弦相似度函数可以使用任意margin-based的loss函数，在这里我们使用的是ArcFace. 如Figure 1所示，困难样本的负例余弦相似度调制系数$I(t, \theta_j)$同时依赖于t和$\theta_j$. 在训练的前期，从简单样本中学习对模型收敛较为有利。所以t应该靠近0，从而$I(\cdot)=t+\cos \theta_j$小于1.因此困难样本的权重得到削减、简单样本得到相应的加强。随着训练过程进行，模型逐渐更加关注困难样本，即t的值应该增加使$I(\cdot)$大于1. 导致困难样本获取更大权重，得以增强。而且在同一个训练阶段中，$I(\cdot)$随$\theta_j$单调递减，因此困难样本能够根据其难度被赋予更大的系数。t的值是自动确定的。
**Optimization.** 在计算梯度的时候，困难样本的梯度主要受$M(\cdot)=2\cos \theta_j +t$的影响，它由两部分组成：负例余弦相似度$\cos \theta_j$和t的值。如Figure 2所示，一方面系数随着t的增长而增长从而增强困难样本。另一方面，这些系数根据它们的难度$\cos \theta_j$被分配不同的重要性。因此在Figure 2中，M的值在每个训练循环中都呈现一个范围。
**Adaptive Estimation of t.** 为不同的训练阶段确定不同的t值很重要。理想情况下，t的值可以由模型的训练阶段来确定。通过经验发现正例余弦相似度的平均值是一个较好的指示器。但是基于统计的mini-batch方法通常面临一个问题：当在一个mini-batch中有很多比较极端的数据的时候，则统计数据会有大量噪声，从而导致期望不稳定。一般会用指数移动平均值(EMA, Exponential Moving Average)来解决这一问题。令$r^{(k)}$代表第k个batch的正例余弦相似度均值，可以写作$r^{(k)}=\sum_i\cos \theta_{y_i}$，因此有：
$$
t^{(k)}=\alpha r^{(k)} +(1-\alpha) t^{(k-1)},
$$
其中$t^0=0, \alpha$是动量参数，设置为0.99. 使用EMA避免了手动调节超参数，而且使困难样本负例余弦相似度$I(\cdot)$能够适应当前训练阶段。总结起来，CurricularFace的loss函数形式为：
$$
\mathcal{L}=-\log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j=1,j\ne y_i}^n e^{sN(t^{(k)},\cos\theta_j)}}
$$
其中$N(t^{(k)},\cos\theta_j)$在前面已经定义。整个训练过程如Algorithm 1所示。
# When Does Label Smoothing Help

## Abstract

一个多分类神经网络的泛化能力和学习速度通常可以通过使用对hard target进行加权平均得到的soft target和标签的均匀分布而大幅度提高。通过这种方式对标签进行平滑处理可以防止网络过于自信，而且标签平滑已被用于多个sota模型中，包括图片分类、翻译和语音识别。尽管这一技术已被广泛使用，我们对标签平滑的理解仍然很浅薄。本文从经验上展示除了能够提升泛化能力外，标签平滑技术还可以提升模型准确性，从而显著提升beam-serch效果。但是同样观察到如果一个教师网络通过标签平滑训练得到，那么将其通过知识蒸馏得到的学生网络的效果会变差。为了解释这一现象，我们对标签平滑如何改变网络倒数第二层学习的表达进行了可视化。我们展示了标签平滑可以促进训练样本中同一类的样本表达组成更紧凑的聚类族。这导致了不同类别样本的相似度的logits信息的丢失，这一现象会很大程度影响知识蒸馏，但并不损害模型预测的泛化性或准度。



## 1 Introduction

标签平滑，通过在数据集targets平均分布的加权混合target上而非数据集的hard targets上计算交叉熵而提升精度。

标签平滑已成功在多个领域的深度学习模型提升精度，如图像分类、语音识别和机器翻译。

尽管标签平滑是一个广泛使用的用于提升网络性能的技巧，但对其在何种情况下以及何时起作用却知之甚少。本文研究一下通过标签平滑训练的网络，并表述这种网络的一些有趣的特性。

contributions:

* 介绍一种新的可视化方法，基于倒数第二层激活的线性映射。这种可视化方法对是否使用标签平滑进行训练的网络的倒数第二层表达之间的区别给出了直观的理解。
* 演示了标签平滑隐式地对学习模型进行了校准，从而使模型预测的置信度与预测的准确度更一致。
* 展示了标签平滑技术会损害知识蒸馏，即当教师模型通过标签平滑技术训练，则学生模型表现会变差。进一步展示了这一反面影响是由于logits信息丢失而造成的。

### 1.1 Preliminaries

标签平滑的数学描述。假设将一个神经网络的预测写作其倒数第二层激活的一个函数$p_k=\frac{e^{x^Tw_k}}{\sum_{l=1}^Le^{x^Tw_l}}$，其中$p_k$是模型指定给第$k$个类别的似然，$w_k$代表最后层的权重和偏置，$x$是倒数第二层的向量，接了个1作为偏置。对于使用hard targets训练的网络，我们最小化真实目标$y_k$和网络输出$p_k$间交叉熵的期望值，表示为$H(y, p)=\sum_{k=1}^K-y_k\log{p_k}$，其中对于正确类别$y_k$为1，其余为0. 对于使用参数为$\alpha$的标签平滑进行训练的网络，我们最小化修改后目标$y_k^{LS}$和网络输出$p_k$之间的交叉熵，其中$y_k^{LS}=y_k(1-\alpha)+\alpha/K$.



## 2 Penultimate layer representations

使用标签平滑训练一个网络会促使正确类别和错误类别的logit间差异向一个依赖于$\alpha$的常数靠拢。作为对比，使用hard targets训练的网络通常会导致正确类的logit比其余错误类的logit大很多，同时也允许错误类logits间差异很大。直观上理解，第$k$个类别的logit$x^Tw_k$可被视为倒数第二层激活$x$和模板$w_k$间欧氏距离平方的一个度量，因为$\lVert x-w_k \lVert^2=x^Tx-2x^Tw_k+w_k^Tw_k$. 这里每一类都有一个模板$w_k$，当计算softmax输出的时候不考虑$x^Tx$，而且不同类别的$w_k^Tw_k$通常是常数。因此，***标签平滑会促使倒数第二层激活向正确类别模板靠近，同时与错误类别的距离相等***。为了观察标签平滑的这一特性，提出一种可视化方案，基于下述步骤：（1）选三个类别，（2）找一个这三类模板所在平面的正交基，（3）将这三类样本的倒数第二层激活映射到这个平面上。这个可视化方法在2D上展示了激活如何围绕模板聚集和标签平滑如何强制样本和其他类聚类族间产生距离架构。

![Figure 1](1.png"Figure 1")

![Table 2](t2.png"Table 2")

Figure 1中展示了在CIFAR-10，CIFAR-100和ImageNet上训练的图像分类器的倒数第二层表示的可视化结果，架构分别为AlexNet，ResNet-56和Inception-v4. Table 2展示了这些模型使用标签平滑的效果。首先从描述CIFAR-10中三个类别（飞机，汽车和鸟）的可视化结果开始。前两列代表不使用标签平滑的训练和验证集。观察到投影分散到定义明确但是范围较大的聚类族中。后两列展示了用因子为0.1的标签平滑进行训练的网络。可以观察到这种情况下的聚类族更紧凑，因为标签平滑会促使训练集中的每个样本与其他类别模板变成等距的。Therefore, when looking at the projections, the clusters organize in regular triangles when training with label smoothing, whereas the regular triangle structure is less discernible in the case of training with hard-targets (no label smoothing). Note that these networks have similar accuracies despite qualitatively different clustering of activations.

第二行中，


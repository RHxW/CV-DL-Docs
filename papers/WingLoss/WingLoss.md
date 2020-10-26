# Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks

## Introduction

人脸关键点定位，或称人脸对齐，目标是从2D图像上找到一组预定义的关键点坐标。一个面部关键点通常有明确的语义意义，比如鼻尖或瞳孔，为其他面部分析任务（如人脸识别，情绪估计，3D人脸重建等）提供了大量的几何信息。

限制场景下人脸关键点定位精度已经很高，难点是非限制场景下鲁棒的高精度关键点定位任务。其影响因素有很多，比如姿态，表情，光照，模糊和遮挡等。

深度学习的一个关键方向是定义一个loss函数，使获得更好学习表示的能力。大多数现有的人脸关键点定位使用L2 loss。但是L2 loss对异常值较敏感，在Fast R-CNN算法中也提到过这个问题，可以将L2 loss 替换为L1 loss 来缓解这个问题。为了进一步减轻这一影响，提出新的loss函数，称为Wing loss，如Figure 1

![avatar](1.png"Figure 1")



## CNN-based facial landmark localisation

CNN-based关键点定位的目标是找到一个非线性映射：
$$
\Phi:\mathcal{I}\to \mathrm{s},\quad\quad\quad(1)
$$
这个映射可以根据一个给定的彩色图片$\mathcal{I}\in\mathbb{R}^{H\times W\times 3}$输出一个向量$\mathrm{s}\in\mathbb{R}^{2L}$.输入图片通常是根据人脸检测输出的边界框裁切的图片。向量的形式$\mathrm{s}=[x_1,...,x_L,y_1,...,y_L]^T$，其中$L$是2D人脸关键点的数量，$(x_l,y_l)$是第$l$个关键点的坐标。

给定一组标注好的训练样本$\Omega=\{\mathcal{I}_i,\mathrm{s}_i\}_{i=1}^N$，CNN训练的目标是找到一个$\Phi$使下式最小：
$$
\sum\limits_{i=1}^Nloss(\Phi(\mathcal{I}_i),\mathrm{s}_i),\quad\quad\quad(2)
$$
其中$loss()$是一个预定义的loss函数，用来衡量人脸形状预测值和真实值间差异。这样的例子中，CNN作为一个回归器以有监督的方式训练。



## Wing loss

设计一个合适的loss函数对于CNN-based面部关键点检测来说至关重要。但是现有的很多深度学习方法的人脸关键点预测系统采用L2 loss，本文展示L1 loss 和smooth L1 loss要比L2 loss 效果好很多，并提出一个新loss函数称为Wing loss，进一步提升关键点预测的精准度。

### Analysis of different loss functions

给定训练图片$\mathcal{I}$和网络$\Phi$，可以预测面部关键点向量$\mathrm{s'}=\Phi(\mathcal{I})$.其loss函数定义为：
$$
loss(\mathrm{s,s'})=\sum\limits_{i=1}^{2L}f(s_i-s_i'),\quad\quad\quad(3)
$$
其中$\mathrm{s}$是gt关键点向量。上式中的$f(x)$，L1 loss使用$L1(x)=|x|$，L2 loss使用$L2(x)=\frac{1}{2}x^2$.smooth L1 loss的分段形式为：
$$
smooth_{L1}(x)=
\begin{cases}
\frac{1}{2}x^2 \quad\quad\quad \mathrm{if}|x|<1 \\
|x|-\frac{1}{2} \quad\quad \mathrm{otherwise}
\end{cases},
\quad\quad\quad(4)
$$
它对于小的$|x|$值为二次项，对于大的$|x|$值为线性。具体来说，smooth L1在$x\in(-1,1)$上使用$L2(x)$，其他地方变为$L1(x)$.Figure 3展示了这几个函数的图像。

![avatar](3.png"Figure 3")

关键点检测广泛使用L2 loss，但是L2 loss广为人知的特点是对异常值敏感。这也是为什么后来将L2 loss 优化为L1 loss的原因。

### The proposed Wing loss

L1和L2loss的梯度大小分别为1和$|x|$，对应的优化步长量级应该为$|x|$和1.在这两种情况下最小化是比较简单的。但是当同时对多个点进行优化时情况就变得很复杂，如公式(3).这两种方案的loss都会被较大的误差做决定。对于L1来说，所有点的梯度重要性都一样，但是更新步长会被较大的误差不成比例地影响。对于L2，步长一样但是梯度由较大误差决定。因此这两种loss都难以修正小位移偏差。

小误差的影响可以通过替代损失函数来增强，例如$\ln x$.其梯度是$1/x$，在接近0时变大。优化步长量级为$x^2$.当多个点一起时，loss主要由小误差决定，但是步长由大误差决定。这样修复了不同量级误差影响的平衡性。但是为了防止大步长潜在方向错误，不对小位置误差进行过补偿就很重要。可以通过在log函数上加一个正的补偿项实现。

这种形状的loss函数适合处理较小的位置误差。但是在in-the-wild人脸的关键点检测中我们可能遇到极端姿态，其与初始的关键点位置误差可能很大。这种情况下，loss函数应该能够根据这些大误差值迅速恢复到对应数值。这就需要loss函数行为更像L1或L2.由于L2对异常值敏感，我们倾向于L1.

上述观点直观地描述了这样的一个loss函数，其对于小误差应该像一个带偏置项log函数，对于大误差像L1.这样的复合函数可以被定义为：
$$
wing(x)=
\begin{cases}
w\ln(1+|x|/\epsilon) \quad\quad \mathrm{if}|x|<w \\
|x|-C \quad\quad\quad\quad\quad \mathrm{otherwise}
\end{cases},
\quad\quad\quad(5)
$$
其中非负项$w$将非线性部分的范围设置为$(-w,w)$，$\epsilon$限制了非线性区域的弧度，$C=w-w\ln(1+w/\epsilon)$是一个常量，将线性和非线性两部分光滑地连接到一起。
$$
Wing(y,\hat y)=
\begin{cases}
w\ln(1+|\frac{y-\hat y}{\epsilon}|) \quad\quad \mathrm{if}|(y-\hat y)|<w \\
|y-\hat y|-C \quad\quad\quad\quad \mathrm{otherwise}
\end{cases}
\\
C=w-w\ln(1+w/\epsilon)
$$
其中$y$是gt的heatmap，$\hat y$是预测的heatmap

注意不要将$\epsilon$设置的很小，会引起训练过程非常不稳定，在误差很小的时候出现梯度爆炸。实际上Wing loss的非线性部分只是取了$\ln(x)$在$[\epsilon/w,1+\epsilon/w]$间的曲线，然后再X和Y方向用一个因子$w$来缩放。并且还使用了将$wing(0)=0$的变换强制保持loss函数的连续性。



*****

为了解决关键点定位中关键点位置回归loss对小误差不敏感的特点而设计的loss函数，在L2->L1->SmoothL1的路线上进一步加大小误差对整体的影响
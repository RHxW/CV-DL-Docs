# LARNet Lie Algebra Residual Network for Face Recognition
## Abstract
本文提出一个基于李代数的新方法，探索了人脸在3D空间的旋转如何影响CNN生成的特征。证明了人脸在图片空间的旋转等同于CNN在特征空间的一个额外的残差部分，该部分仅受旋转的影响。基于理论分析提出了LARNet(Lie Algebraic Residual Network)用于解决跨姿态人脸识别问题，LARNet的构成：
1. 一个用于从输入图片中解码旋转信息的残差子网络
2. 一个通过学习旋转量级来控制残差部分对特征学习贡献强度的门控子网络
![Figure 1](1.png 'Figure 1')
## 1. Introduction
尽管一些人脸识别的模型足够健壮，并且泛化能力很强，但是在非限制场景中仍然有跨年龄、跨模态、跨姿态以及遮挡等难题的存在。由于深度模型的泛化能力与训练集的规模相关性较高，如果训练集中正脸和侧脸图片的分布不均衡、侧脸不足，那么深度特征就更倾向于关注正脸，因此学习到的结果会偏向不完整的统计结论。可以通过数据增强的手段构建更大数据集来解决这一问题，一个常见方法是生成侧脸或者将一组图片作为一张输入，从而减轻侧脸数据的需求。另一个方法是合并更多的数据信息包括多任务学习以及模板适应。

我们观察到一个现象：正侧脸的差异是由头部旋转造成的，而这个原因不应该在侧脸识别中被忽视掉。但是在CNN中加入旋转矩阵比较困难，因为旋转矩阵在乘法下是closed，加法下则不是，而梯度下降的时候加法操作很多。受SLAM(simultaneous localization and mapping)中的姿态估计启发，开发了一个利用李代数来实现更新CNN中旋转矩阵的方法。
我们证明了每一个正侧脸样本对都由一次旋转操作联系在一起，它们对应的深度特征同样保留着对应的旋转关系。基于这一结论，提出李代数残差网络LARNet(Lie Algebra Residual Network)，实现了人脸旋转在特征空间的渲染，如果Figure 1所示。本文主要贡献：
1. 理论上证明了在使用李代数的残差网络中，正侧脸样本对的特征存在一种基于旋转的物理上的关系，而且相当于在CNN的特征空间上的一个额外的残差部分
2. 设计了一个新的门控子网络，既不需要修改已有backbone结构，也不依赖大量模块，但是能够带来较大的性能提升
3. LARNet增强了模型在特征表达和分类上的能力

## 2. Related Work
...

## 3. Methodology
首先假设正脸图像和其侧脸图像在原始的3D空间有对应的旋转关系。简单起见，只讨论正交关系的旋转。

### 3.1. Problem Formulation
目标是找到输入正脸图像和期望的侧脸图像的特征间的变换关系，以至于能在特征空间实现‘正面化’并获取一个对姿态变换鲁棒的特征表达，如Figure 1所示。
用F(x)代表CNN提取特征的函数，x代表输入图像。对x中的每一个像素(u,v)采用齐次坐标表示$(u,v,1)^T$, 为了方便起见，仍用x表示这些3维的齐次坐标

用d代表某一层的维度。则提取的特征$F(x)\in \mathbb{R}^d$. 需要证明存在一种映射$\mathcal{R}_{map}(\cdot):\mathbb{R}^d \rightarrow \mathbb{R}^d$可以作为图像空间旋转$\textbf{R}\in SO(3)$在特征空间中的等价:
$$
F(\textbf{R}\cdot x)=\mathcal{R}_{map}(F(x)) \qquad(1)
$$
对于正脸图像$x_f$和对应的侧脸图像$x_p$, 它们的单应性变换矩阵(homography transformation matrix)退化成一个旋转矩阵:$x_f=\textbf{R}\cdot x_p$，因此有：
$$
F(x_f)=F(\textbf{R}\cdot x_p)=\mathcal{R}_{map}(F(x_p)) \qquad(2)
$$
(上式的意思是在图片空间旋转后的人脸提特征等价于某种特征空间的变换)
进一步尝试使用李群理论证明映射$\mathcal{R}_{map}(\cdot)$可以分解成残差部分相加的形式，且仅由旋转确定：
$$
F(x_f)=F(x_p)+\omega (\textbf{R})\cdot \textbf{C}_{res}(\textbf{R},x_p) \qquad (3)
$$
因此，只需要一个残差子网络$\textbf{C}_{res}$用于从输入的人脸图片中解码姿态变化信息，以及一个用于学习旋转量级的门控子网络$\omega$从而对特征学习过程的残差部分的强度实现控制。上面这个公式是LARNet的核心思想。

### 3.2. Rotation in Networks and Lie Algebra
（一堆证明blahblahblah）

在之前曾提到两张图片$x_p$和$x_f$间的单应性关系是通过旋转联系起来的，但是通常来说，CNN无法保证这种关系。但是经过证明，发现这种关系在每层的梯度下降过程中得以继承下来。实际上，由于$\textbf{R}\in SO(3)$,所以$\textbf{R}\cdot x_L^p$和$x_L^f$是渐近稳定的（根据李雅普诺夫第二方法）。随着ResNet的训练过程的逐渐进行，$\textbf{R}\cdot x_L^p$和$x_L^f$的特征有相同的收敛表示：$F(x_f)=F(\textbf{R}\cdot x_p)$. 而且对旋转关系进行分解，令$V_{res}=F(\textbf{R}\cdot x_p)-\mathcal{R}_{map}F(x_p)\in \mathbb{R}^d$为残差向量，则有：
$$
\mathcal{R}_{map}^{-1}F(x_f)=F(x_p)+\mathcal{R}_{map}^{-1} \cdot V_{res}, \\
F(x_f)=F(x_p)+\mathcal{R}_{map}^{-1}(V_{res}+\mathcal{R}_{map}F(x_f)-F(x_f))
$$
由于在训练阶段，$F(x_p)$会向$\mathcal{R}_{map}F(x_f)$靠拢，因此上式变为：
$$
F(x_f)\approx F(x_p)+\mathcal{R}_{map}^{-1}(F(x_p)-\mathcal{R}_{map}F(x_p))
$$
这与公式3相吻合。因此可以设计一个门控函数$\omega(\textbf{R})$作为$\mathcal{R}_{map}^{-1}$用于过滤特征流，$\textbf{C}_{res}(\textbf{R},x_p)=F(x_p)-\mathcal{R}_{map}F(x_p)$部分可以通过残差网络训练得到。

### 3.3. The Architecture of Subnet
需要设计一个残差子网络$\textbf{C}_{res}$来实现对输入图片中人脸的姿态变化信息的解码。公式3展示了用简单网络结构从干净的特征学习的可能。最简单的方式是在现有的backbone上增加一个门控残差项。可以安排在最终FC层之前而不需要修改任何权重。所以我们的结构有两个FC层以及PReLU作为激活函数。用侧脸特征$F(x_p)$和旋转后的正脸特征$\mathcal{R}_{map}F(x_f)$间的L2正则来训练：
$$
\min_{\Omega_p} \sum \lVert F(x_p)-\mathcal{R}_{map}(\Omega_p)F(x_f)\rVert_2^2
$$
其中$\Omega_p$代表可学习的参数。用正侧脸样本对来训练这个子网络。如果采用复杂结构，则可能增加过拟合的风险，而两个FC层的设计即考虑到任务难度，也兼顾了模型鲁棒性。
进一步设计了门控函数$\omega$，它需要满足下列条件：
* $\omega \in [0,1]$. 直观上，当输入是正脸图片$x_0$时，同一个网络的特征表示几乎没有差异，那么$\textbf{C}_{res}$会引入误差从而削弱分类的能力。因此此时需要门控函数为0. 理想状况下当侧脸时（$F(x_0)-F(x_{\pi /2})$）该值最大，因此最大值为1：
  $$
  F(x_0)=F(x_{\pi /2})+1*(F(x_0)-F(x_{\pi /2}))=F(x_0)
  $$
* $\omega$有对称的权重。门控函数学习旋转的量级，用来控制残差模块影响特征的程度，相同的旋转角度应该对正脸图像有相同的的影响。同时会在训练中使用翻转的数据增强方法来增强模型的对称性。

而且，翻转角、偏航角和俯仰角对人脸识别的影响是不同的。翻转角在人脸对齐之后可以忽略。较大俯仰角的图片很稀少。经过总结，可得$\omega=|\sin \theta|$，令$\sin \theta=\lVert (\sin_{pitch},\sin_{yaw}, \sin_{roll}) \rVert_{\infty}$对所有角度介于$[-\pi/2,\pi/2]$之间，这样保证了在李代数$\Phi$和旋转$\textbf{R}$间存在一个一对一的对应关系，同时也保证了所提出理论的完整性。
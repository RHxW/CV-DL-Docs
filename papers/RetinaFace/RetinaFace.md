# RetinaFace: Single-stage Dense Face Localisation in the Wild

利用额外监督（extra-supervised）和自监督（self-supervised）联合起来的多任务学习实现多尺度人脸的像素级人脸定位。

后续添加了一个自监督的网格解码器（mesh decoder）分支来预测面部3D形状（与已有的分支平行）

使用轻量级backbone网络，RetinaFace可以在cpu上跑VGA分辨率视频图像



## Introduction

人脸定位的狭义定义可以理解为传统人脸检测，即在没有尺度和位置先验的条件下对人脸边界框的估计。但是本文提出一个人脸定位的广义定义，包括人脸检测，人脸对齐，像素级人脸解析和3D密集相似回归。这种密集人脸定位方法可以在不同尺度下提供准确的面部位置信息。

与一般的物体检测不同，人脸检测的特点是长宽比的取值范围较小（从1:1到1:1.5），但是尺度范围很大（从几个像素到几千像素）。通过使用多任务损失（ coming from strongly supervised and self-supervised signals），我们提出一个SOTA的密度人脸定位方法。

![avatar](/1.png"Figure 1")

典型的人脸检测训练过程包括分类和box回归loss

在Mask R-CNN中，在已有的边界框识别和回归分支的基础上，通过加入一个与前者平行的预测物体mask的分支，显著地提升了检测的效果。这个效果证明了密度像素级标注信息对检测效果同样有提升。

在FAN中，提出一个anchor级的注意力图来提升遮挡人脸的检测效果。但是这个注意力图粒度很粗而且不包含语义信息。最近，自监督的3D可变形模型在自然环境下的3D人脸建模方向取得了不俗的进展。特别是Mesh Decoder方法通过在形状和纹理上使用图卷积的方式达到了超实时的速度。

但是在单阶段分类器上使用mesh decoder的难点是：

1. 相机参数难以准确估计
2. 隐含的形状和纹理表示是通过一个特征向量预测的（来源于特征金字塔的$1\times1$卷积）而非RoI pooled特征，有特征偏移的隐患。

本文在预测像素级3D人脸形状的自监督学习过程中引入一个与现有监督学习分支平行的mesh decoder分支。



## Related Work

* **Image pyramid v.s. feature pyramid**：滑动窗口的模式，在一个图像网格上使用一个分类器的方法可以追溯到几十年前。之后特征图金字塔出现，在多尺度特征图上的滑动anchor迅速统治了人脸检测领域。
* **Two-stage v.s. single-stage**：目前的人脸检测方法继承了一些一般物体检测方法的成果，可以分为2大类：二阶段方法（如：Faster R-CNN）和一阶段方法（如：SSD和RetinaNet）。二阶段方法的机制是"propoasl and refinement"，其特点是高定位准确率。与之相对的单阶段方法密集地对人脸位置和尺度进行采样，这样做的结果是在训练时会引起很严重的正负样本不平衡问题。为了解决这个不平衡问题，广泛使用了采样加权法。与二阶段方法相比，单阶段方法更高效且有更高的召回率，但同时也有更高假阳率的风险，而且定位准确率上也不如二阶段方法。
* **Context Modelling**：为了增强模型在捕捉小脸上的上下文推理能力，SSH和PyramidBox在特征图金字塔上使用了上下文模块在欧几里得网格（Euclidean grids）上增大感受野。为了增强CNN对柔性变换（non-rigid transformation）的建模能力，形变卷积网络（deformable convolution network，DCN）使用一个形变层来对几何变换进行建模。WIDER Face 2018的冠军方法指出刚性（expansion，扩张）和柔性（deformation，形变）的上下文建模对提升人脸检测效果是互补（complementary）和正交（orthogonal）的。
* **Multi-task Learning**：由于对齐的人脸形状能够为人脸分类提供更高质量的特征，人脸检测和对齐的联合算法应用广泛。在Mask R-CNN中，增加与已有分支平行的mask预测分支显著地提升了检测效果。

![avatar](/2.png"Figure 2")

## RetinaFace

### Multi-task Loss

对任意训练anchor $i$，最小化如下多任务损失：
$$
L=L_{cls}(p_i,p_i^*)+\lambda_1p_i^*L_{box}(t_i,t_i^*)\\
+\lambda_2p_i^*L_{pts}(l_i,l_i^*)+\lambda_3p_i^*L_{pixel}.
\quad\quad\quad(1)
$$
(1)人脸分类损失$L_{cls}(p_i,p_i^*)$，其中$p_i$是anchor $i$是人脸的预测值，$p_i^*$是label（0，1）.分类损失$L_{cls}$使用二分类（人脸/非人脸）softmax 损失函数

(2)人脸边界框回归损失$L_{box}(t_i,t_i^*)$，其中$t_i=\{t_x,t_y,t_w,t_h\}_i$和$t_i^*=\{t_x^*,t_y^*,t_w^*,t_h^*\}_i$分别代表与正anchor相关的box坐标预测值和gt值。采用Fast R-CNN的方式归一化box的回归目标值（中心位置，宽和高），$L_{box}(t_i,t_i^*)=R(t_i-t_i^*)$，其中$R$为Fast R-CNN中定义的鲁棒的损失函数（smooth-L1）

(3)面部关键点回归损失$L_{pts}(l_i,l_i^*)$，其中$l_i=\{l_{x_1},l_{y_1},...,l_{x_5},l_{y_5}\}_i$和$l_i^*=\{l_{x_1}^*,l_{y_1}^*,...,l_{x_5}^*,l_{y_5}^*\}_i$分别代表与正anchor相关的五个预测关键点和gt关键点。与box中心回归相似，关键点回归也采用基于anchor中心的归一化方法。

(4)密度回归损失$L_{pixel}$（见公式3）

损失平衡参数$\lambda_1-\lambda_3$分别设置为0.25，0.1和0.01，意味着通过监督信号增加box和关键点的重要性

### Dense Regression Branch

* **Mesh Decoder.** 直接使用现成的mesh decoder（mesh convolution和mesh up-sampling），是一个基于快速局部谱滤波的图卷积方法。为了进一步加速，还使用了一个形状纹理联合decoder

  ![avatar](/3.png"Figure 3")

  下面简要介绍一下图卷积的概念并指出为什么可以用作快速解码。如Figure 3(a)所示，一个2D的卷积操作是在欧几里得网格感受野上的“带权重的核相邻求和”操作。图卷积的概念与之类似，如Figure 3(b)所示。但是图上的相邻距离为两个顶点间的最少边数量。定义一个有颜色的面部网格（coloured face mesh）为$\mathcal{G=(V,E)}$，其中$\mathcal{V}\in \mathbb{R}^{n\times6}$，是包含结点形状和纹理信息的一组面部顶点的集合。$\mathcal{E}\in \{0,1\}^{n\times n}$是一个稀疏的临近矩阵，它编码了图中各顶点的连接状态。图拉普拉斯矩阵定义为$L=D-\mathcal{E}\in\mathbb{R}^{n\times n}$，其中$D\in\mathbb{R}^{n\times n}$是一个对角阵，有$D_{ii}=\sum_j\mathcal{E_{ij}}$

  卷积核为$g_{\theta}$的图卷积可以表示为一个$K$阶的递归的切比雪夫多项式：
  $$
  y=g_{\theta}(L)x=\sum\limits_{k=0}^{K-1}\theta_kT_k(\tilde{L})x,
  \quad\quad\quad(2)
  $$
  其中$\theta\in\mathbb{R}^K$是一个切比雪夫系数向量，$T_k(\tilde L)\in \mathbb{R}^n$是$k$阶切比雪夫多项式在缩放后的拉普拉斯矩阵$\tilde L$求的值。

  令$\bar x_k=T_k(\tilde L)x\in \mathbb{R}^n$，可以递归地计算$\bar x_k=2\tilde L\bar x_{k-1}-\bar x_{k-2},\bar x_0=x,\bar x_1=\tilde Lx$.

  整个操作（filtering operation）非常高效，包括$K$个稀疏矩阵-向量乘法和一次稠密矩阵-向量乘法$y=g_{\theta}(L)x=[\bar x_0,...,\bar x_{K-1}]\theta$.

* **Differentiable Renderer.** 得到预测的形状和纹理参数$P_{ST}\in \mathbb{R}^{128}$之后，使用一个高效的可微分3Dmesh renderer来将一个有颜色的mesh$\mathcal{D}_{P_{ST}}$投影到2D平面，相机参数为$P_{cam}=[x_c,y_c,z_c,x_c',y_c',z_c',f_c]$（相机位置，相机姿态和焦距focal length），光照参数为$P_{ill}=[x_l,y_l,z_l,r_l,g_l,b_l,r_a,g_a,b_a]$（光源位置，色值（color values，色彩明度）和环境光颜色）

* **Dense Regression Loss.** 获取渲染后的2D面部图像$\mathcal{R}(\mathcal{D}_{P_{ST}},P_{cam},P_{ill})$，使用下述函数对渲染图和原图的像素差异进行比较：
  $$
  L_{pixel}=\frac{1}{W*H}\sum\limits_i^W\sum\limits_j^H
  \lVert\mathcal{R}(\mathcal{D}_{P_{ST}},P_{cam},P_{ill})_{i,j}-I_{i,j}^*\rVert_1,
  \quad\quad\quad (3)
  $$
  其中$W$和$H$是anchor裁切$I_{i,j}^*$的宽和高。


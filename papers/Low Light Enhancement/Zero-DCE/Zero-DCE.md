# Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation

## Abstract
本文提出一个称为Zero-Reference Deep Curve Estimation(Zero-DCE)的图像增强方法，将图像增强视为使用深度神经网络实现的单张图像曲线估计任务。本方法训练一个轻量级网络DCE-Net用来估计给定图像的像素级和次动态调整曲线。曲线估计的设计考虑到了像素值的范围、单调性和可微性。Zero-DCE的特点在于它不需要任何参考，也就是说训练的时候不需要成对的数据，而这个特点是通过一系列无参考loss函数实现的。

## 1. Introduction
拍照的时候经常会遇到各种光照问题，例如环境光照的不充分和不平衡情况、背景光太强和曝光不足等问题。

本文提出一个基于深度学习的低光照图片增强新方法，无参考深度曲线估计(Zero-Reference Deep Curve Estimation, Zero-DCE). 这种方法可以处理包括低光照和不均匀光照在内的多种光照场景。该方法采用估计图像曲线代替图像间转换任务。具体来说，接收一张低光照图片作为输入，然后生成高次曲线，然后用这些曲线对像素值动态范围进行调整，调整后的图像作为增强图像输出，它保留了增强图像的范围以及相邻像素的对比度。

## 2. Related Work
...

![Figure 2](2.png 'Figure 2')

## 3. Methodology
Zero-DCE的架构如Figure 2所示。用深度曲线估计网络(Deep Curve Estimation Network, DCE-Net)来估计一张图片的一组最佳拟合效果的光照增强曲线(Light-Enhancement curves, LE-curves). 然后迭代地用曲线对原图进行修正，最后得到增强后的图像。

### 3.1 Light-Enhancement Curve
受图像编辑软件曲线调整启发，设计了一种能够自动将低光照图映射到增强图的曲线，其中自适应曲线的参数仅依赖于输入图片，这种曲线的设计需要满足三个条件：
1. 增强后每个像素点的值要在归一化的0到1之间，目的是为了防止由于截断引起的信息丢失
2. 为了保留相邻像素的差异（对比度），该曲线应该是单调的
3. 曲线的形式应该尽可能的简单并且可微

根据上述要求，设计了二次曲线，表示为：
$$
LE(I(x);\alpha)=I(x)+\alpha I(x)(1-I(x)), \qquad (1)
$$
其中x代表像素的坐标，$LE(I(x);\alpha)$是输入I(x)的增强结果，可训练曲线参数$\alpha\in [-1,1]$用于调整LE曲线的量级，同时控制曝光度。输入的像素值都归一化到0，1之间，而且所有操作都是针对像素点的。将LE曲线分别作用于RGB通道，而不是只作用于光照通道，因为三通道的调整可以更好地保留内在颜色并减少过饱和的风险。

**Higher-Order Curve.** 公式1中定义的LE曲线可以迭代使用来进行多种调整：
$$
LE_n(x)=LE_{n-1}(x)+\alpha_nLE_{n-1}(x)(1-LE_{n-1}(x)), \qquad (2)
$$
其中n是迭代次数，它控制曲率。本文的方法将n设定为8，足以较好地处理大多数情况。当n=1的时候，公式2就退化为公式1. Figure 2(c)展示了不同$\alpha$和n高次曲线，这种高次曲线能够提供比Figure 2(b)更强的校正能力。

![Figure 3](3.png 'Figure 3')

**Pixel-Wise Curve.** 和一次曲线相比，高次曲线能够在更广的动态范围中对图片进行调整。尽管如此，这种方式仍然是一种全局的调整方法，因为参数$\alpha$作用于全部像素之上。全局的映射方法在局部会有过增强或欠增强的倾向。为了解决这一问题，将参数$\alpha$设定为像素级参数，即每个像素都有一个对应的曲线和最优拟合的$\alpha$用来调整动态范围，因此公式2变为：
$$
LE_n(x)=LE_{n-1}(x)+\mathcal{A}_nLE_{n-1}(x)(1-LE_{n-1}(x)), \qquad (3)
$$
其中$\mathcal{A}$是和输入图像尺寸一致的参数图。此处假设局部区域内的像素都有相同的强度（以及相同的调整曲线），因此输出结果中相邻的像素仍然保留单调的关系。这样，像素级高次曲线仍满足之前提到的三个条件。

在Figure 3中展示了一个估计曲线参数图的例子，其中不同颜色通道的最佳拟合参数图有相似的校正趋势但是具体值不同，体现了低光照图片三通道之间的联系和差别。曲线参数图精确地展现了不同区域的亮度，根据拟合图，就可以通过调整曲线的方式得到最终结果。如Figure 3(e)中所示，增强后的图片展现了暗光区域的细节并保留了较亮区域的信息。

### 3.2 DCE-Net
提出深度曲线估计网络(Deep Curve Estimation Network, DCE-Net), 其架构如Figure 4中所示。
![Figure 4](4.png 'Figure 4')

### 3.3 Non-Reference Loss Functions
为了实现无参考学习，提出一组可微的无参考loss函数，用于评估增强后图片的质量，下面介绍的四种loss是DCE-Net的训练中采用的loss函数。

**Spatial Consistency Loss.** 空间一致性loss用于保证增强后图片局部区域的一致性：
$$
L_{spa}=\frac{1}{K}\sum_{i=1}^K\sum_{j\in \Omega(i)}(|Y_i-Y_j|-|I_i-I_j|)^2, \qquad(4)
$$
其中K是局部区域的个数，$\Omega(i)$是当前区域的四个相邻区域（上下左右），Y和I分别代表增强后和原图局部区域的平均强度。实验中将局部区域的尺寸设定为$4\times 4$，如Figure 5中所示。
![Figure 5](5.png 'Figure 5')

**Exposure Control Loss.** 为了抑制欠曝光和过曝光区域，设计了曝光控制loss用于控制曝光等级。这个loss衡量区域平均强度与正常曝光E的距离，将E设定为RGB空间的灰度，值设置为0.6，则loss可以表示为：
$$
L_{exp}=\frac{1}{M}\sum_{k=1}^M|Y_k-E|, \qquad(5)
$$
其中M代表尺寸为$16\times 16$的不重合的局部区域

**Color Constancy Loss.** 根据灰度空间颜色一致性假设，即每个通道在整张图片的范围内的平均值趋向于灰色这一假设，设计了色彩一致性loss并构建了校正后三通道之间的关系：
$$
L_{col}=\sum_{\forall (p,q)\in \varepsilon}(J^p-J^q)^2, \varepsilon=\{(R, G),(R, B), (G, B)\}, \qquad (6)
$$
其中$J^p$代表增强后图片p通道的平均强度值。

**Illimination Smoothness Loss.** 为了保留相邻像素间的单调性值，为每个曲线参数图$\mathcal{A}$加入一个光照平滑loss：
$$
L_{tvA}=\frac{1}{N}\sum_{n=1}^N \sum_{c\in \xi}
$$
# SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation

Soft Stagewise Regression Network（SSR-Net），用于单张图片（人脸）年龄估计，模型紧凑

受DEX启发，将年龄估计问题用多分类+分类结果回归的方式解决

SSR-Net使用由粗到细的策略实现多阶段（multi-stage）的多分类（multi-class）任务。每一阶段只负责优化上一阶段的结果来提高年龄估计的精度。因此每一阶段都只有几个类别，也只需要几个神经元，这就大大降低了模型的尺寸。

为了解决年龄分组所带来的量化问题，SSR-Net给每个年龄类别分配一个动态范围（dynamic range），这个动态范围可以根据输入的人脸图片变化。



直观上讲，将年龄估计问题视为一个回归问题好像比离散的分类问题更合理，毕竟年龄是一个连续值。

但是回归形式的年龄估计方法由于其处理过程的随机性和面部特征与真实年龄间映射的不确定性会导致过拟合的出现。

另一方面，人们的年龄可以被轻松地分成几组，比如青少年，中年和老年。因此，这方面的很多研究都采用将年龄量化成不同组的形式把年龄估计问题转化为多分类问题。而这样做要考虑到年龄的分类之间是有顺序、有关联的，并不像普通分类一样类间无关联。因此，将年龄量化分类会遇到量化误差和组间界限模糊的问题。



### Soft Stagewise Regression Network

在单一人脸的年龄估计问题里，训练集是人脸图像$X=\{x_n|n=1..N\}$，每张图片人脸$x_n$的实际年龄是$y_n\in Y$，其中$N$是图片数量，$Y$是年龄区间。

目标是找到一个函数$F$，可以对于给定图像$x$预测其年龄$\tilde{y}=F(x)$

训练的时候使用最小化MAE的方式找到函数$F$

$J(X)=\frac{1}{N}\sum\limits_{n=1}^{N}|\tilde y_n - y_n|$                    (1)



在此之前的工作将年龄估计从回归问题转化为分类问题，例如DEX（DEX: Deep EXpectation of apparent age from a single image）

DEX将年龄区间$Y=[0, V]$划分为$s$个独立的（不重叠的）bin，则每个bin的宽度就是$\frac{V}{s}$

我们将第$i$个bin的特征年龄表示为$\mu_i$，则DEX中$\mu_i=i(\frac{V}{s})$

DEX训练一个$s$类的分类网络，对于给定的图像$x$，网络输出一个分布向量$\vec{p}=(p_0,p_1,...,p_{s-1})$，这个向量代表$x$属于每一类的概率。年龄的预测值为：

$\tilde y=\vec{p}\cdot\vec{\mu}=\sum\limits_{i=0}^{s-1}p_i\cdot\mu_i=\sum\limits_{i=0}^{s-1}p_i\cdot i(\frac{V}{s})$

为了使估计更准确，DEX把年龄区间分得很细，一个bin的宽度为一年。这导致最终阶段的全连接层参数非常多，很耗内存。

在不过多牺牲准确率的前提下要降低模型尺寸，我们提出使用多阶段预测的由粗到细的策略。

假设有$K$个阶段，一个有$s_k$个bin

对于每个阶段都训练一个网络$F_k$，这个网络会生成这个阶段的分布$\vec{p}^{(k)}=(p_0^{(k)},p_1^{(k)},...,p_{s_k-1}^{(k)})$

stagewise regression会按照下述公式预测年龄：

$\tilde y=\sum\limits_{k=1}^{K}\vec{p}^{(k)}\cdot\vec{\mu}^{(k)}=\sum\limits_{k=1}^{K}\sum\limits_{i=1}^{s_k-1}p_i^{(k)}\cdot i\Bigg(\frac{V}{\prod_{j=1}^{k}s_j}\Bigg)$                  (3)

这个公式中的最后一项是第$k$阶段第$i$个bin的宽度$w_k=\frac{V}{\prod_{j=1}^{k}s_j}$

举例：

> 假设要预测的年龄范围是0~90（$V$=90），并假设有两个阶段（$K$=2），每个阶段有三个bin（$s_1=s_2=3$）。
>
> 从分类的角度来看，stage #1 将图片分成青年（0~30），中年（30~60）和老年（60~90）.对于stage #2 ，stage #1中的每个bin会进一步被分成$s_2=3$个bin。因此，stage #2 的bins的宽度就是$\frac{90}{3\cdot 3}=10$.
>
> stage #2的分类器将图像分为相对年轻（+0~10），中间年龄（+10~20）和相对年老（+20~30），各自对应到stage #1的分类。
>
> 需要注意的是，stage #2 只有一个分类器，由stage #1 共用
>
> stage #1 粗粒度地预测年龄，stage #2 细粒度地预测年龄

stagewise regression 的优点是每个阶段的分类数量少，因此模型参数少，模型更紧凑



### Dynamic Range

将年龄区间均匀切分不如使用年龄的模糊和连续性灵活。

这个问题在粗粒度情况下更严重。

我们解决这个问题的方法就是让每个bin都可以根据输入图像动态变化。

修改bin的位移（shift）和尺度（scale）的办法很多，为了能够使用公式3，我们采用修改bin的下标$i$和宽度$w_k$来实现位移和尺度的修正。

对于修正第$k$个stage的bin宽度$w_k$，引入一个$\Delta_k$项来修改$s_k$为$\bar s_k$：

$\bar s_k=s_k(1+\Delta_k)$                   (4)

其中$\Delta_k$是一个回归网络对输入图片的输出。

在修改完$s_k$之后，bin的宽度变为：

$\bar w_k=\frac{V}{\prod_{j=1}^k\bar s_j}$

可见，修改$s_k$可以有效修改bin宽度



对于bin的位移，在每个bin的下标上加上一个偏移项$\eta$

第$k$阶段有$s_k$个bin，因此需要一个偏移向量，$\vec \eta^{(k)}=(\eta_0^{(k)},\eta_1^{(k)},...,\eta_{s_k-1}^{(k)})$

偏移向量也是一个回归网络的输出（输入为图片）

bin的下标$i$被修改为：$\bar i=i+\eta_i^{(k)}$          (6)

下标$\bar i$修改了bin的位置



对于bin的两项修改都是输入图片的回归值

这个与输入相关的动态范围根据输入图像提供了更精确的refinement



## Network Structure

![figure1](1.png"Figure 1")



Figure1(a)展示了SSR-Net的整体网络结构。使用2-stream模型，有两个异构的stream。对于两个stream，基础结构block由3x3卷积，BN，非线性激活函数和2x2池化组成。但是它们采用的激活函数（ReLU vs Tanh）和池化方式（平均 vs 最大）不同，因此它们是异构的。通过这种方式，它们可以获取不同的特征，在融合后就能提高效果。

不同层（level）的特征会进入不同阶段（stage），对每个阶段，两个stream的特征会在某层进入一个融合块（fusion block），如Figure1(b).融合块负责生成阶段级（stagewise）输出，分布$\vec p^{(k)}$，偏移向量$\vec \eta^{(k)}$，和缩放因子$\Delta_k$.

在融合块内部，两个stream的特征会首先经过$1\times1$卷积，激活函数和池化层来获取更紧凑的特征。

要获得$\Delta_k$，两个stream的特征图会使用元素逐个相乘（克罗内克积）的方式$\bigotimes$得到一个乘积，这个乘积然后会经过一个全连接层和一个Tanh函数，得到$\Delta_k \in [-1, 1]$.

$\vec p^{(k)}$和$\vec\eta^{(k)}$都是向量，它们更复杂一些，因此特征会在相乘（元素逐个相乘$\bigotimes$）之前各自经过一个附加的预测块，里面是全连接层和激活函数。

因为$\vec p^{(k)}$表示一个分布，因此使用ReLU函数作为其激活函数来获得正值。$\vec\eta^{(k)}$使用Tanh函数获取或正或负的shift。



### Soft Stagewise Regression

给定网络关于输入图片$x$的stagewise输出$\{\vec p^{(k)},\vec\eta^{(k)},\Delta_k\}$和bin的数量$s_k$，$x$的年龄预测值$\tilde y$的计算公式为：
$$
\tilde y=\sum\limits_{k=1}^K\sum\limits_{i=0}^{s_k-1}p_i^{(k)}\cdot\bar i \bigg(\frac{V}{\prod_{j=1}^k\bar s_j}\bigg),\qquad (7)
$$
其中$\bar i$是公式6定义的修改后的bin索引，$\bar s_j$是公式4定义的修正后的bin数量，$V$是要预测的年龄范围

将公式7成为soft stagewise regression，因为bins根据小数修正。通过这种方式将Softness引入bin索引和宽度。

根据公式1中的MAE，最小化预测值$\tilde y$的$L_1$ loss，这就是用于年龄估计的SSR-Net模型。



*****

将年龄估计问题用多分类+分类结果回归的方式解决

使用由粗到细的策略实现多阶段（multi-stage）的多分类（multi-class）任务。每一阶段只负责优化上一阶段的结果来提高年龄估计的精度。因此每一阶段都只有几个类别

使用动态范围（dynamic range）来解决年龄分组所带来的量化问题，这个动态范围可以根据输入的人脸图片变化。
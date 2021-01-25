# Introduction to Graph Neural Networks

# Chapter 1. Introduction

图是一种对一组对象（节点）和他们之间的关系（边）进行建模的数据结构。作为机器学习的一种独特的数据结构，图在节点分类、连接预测和聚类上吸引了注意力。图神经网络（GNNs）是在图域上操作的基于深度学习的方法。由于其有说服力的效果和高度的可解释性，GNN成为了广泛应用于图分析的方法。

## 1.1 Motivations

### 1.1.1 Convolutional Neural Networks

首先GNNs受到了CNNs的影响。对于高表达能力的特征，CNNs能够提取并构成多尺度局部空间特征，在几乎所有机器学习领域都取得了breakthroughs并在深度学习领域掀起一场革命。深入CNNs和图，发现了CNNs的关键：局部连接，共享权重和多层的使用。这些关键点同样在解决图域上的问题中有着重要地位，因为（1）图是最典型的局部连接结构，（2）和传统的谱图理论相比，权重的共享减少了计算量，（3）多层结构是解决层级模式的关键，它可以捕捉不同尺寸的特征。

但是CNNs只能处理传统的欧几里得数据，例如图片（2D grid）和文本（1D sequence），同样也可以被看作是图的实例。因此寻找CNNs在图上的推广就很好理解了。如Figure 1.1所示，难以定义局部卷积滤波器和池化操作，这阻碍了CNN从欧氏域向非欧域的变换。

![Figure1.1](1.1.png"Figure 1.1")

### 1.1.2 Network Embedding

另一个动机来自于图embedding，其学习使用低维向量来表示图的节点、边或者子图。在图的分析中，传统机器学习方法通常依赖于人工设计的特征且受限于其自身的不可扩展性和高成本。这一类方法的缺点主要有两个：1. 编码器节点间不共享参数，也就代表着参数数量随着节点数量线性增长，导致计算成本很高。2. 直接embedding的方法缺少泛化能力，意味着不能解决动态图或推广到新图上。



# Chapter 1. Basics of Math and Graph

## 2.1 Linear Algebra



...

...

...





# Chapter 4. Vanilla Graph Neural Networks

## 4.1 Introduction

在图中，自然会用节点的特征和与之相关的节点来定义它。GNN的目的是学习一个状态embedding $\textbf{h}_v\in \mathbb{R}^s$，它对每个节点的邻居信息进行了编码。状态embedding $\textbf{h}_v\in \mathbb{R}^s$用于生成一个输出$\textbf{o}_v$，例如预测的节点标签的分布。

![Figure4.1](4.1.png"Figure 4.1")

对于Scarselli的文章，Figure 4.1 展示了一个典型的图。原始的GNN模型解决无向齐次图，其中每个节点都有输入特征$\textbf{x}_v$且每条边也可以有其特征。这篇文章使用$co[v]$和$ne[v]$来表示$v$的边和邻居节点的集合。

## 4.2 Model

给定节点的输入特征和边，下面讨论一下模型如何获取节点的embedding $\textbf{h}_v$和输出embedding $\textbf{o}_v$

为了根据输入邻居来更新节点状态，用到一个参数函数$f$，称为局部转移函数，由所有节点共享。为了生成节点的输出，用到一个参数函数$g$，称为局部输出函数。然后可以将$\textbf{h}_v$和$\textbf{o}_v$定义为：
$$
\textbf{h}_v=f(\textbf{x}_v,\textbf{x}_{co[v]},\textbf{h}_{ne[v]}, \textbf{x}_{ne[v]}), \qquad (4.1) \\
\textbf{o}_v=g(\textbf{h}_v,\textbf{x}_v), \qquad (4.2)
$$
其中$\textbf{x}$代表输入特征，$\textbf{h}$代表隐藏状态。

...

...







# Chapter 5.  Graph Convolutional Networks

## 5.1 Spectral Methods

谱方法使用图的谱表达。

### 5.1.1 Spectral Network

卷积操作可以通过计算图的Laplacian矩阵的特征分解在傅里叶域中进行定义。这个操作可以定义为信号（每个节点的标量）$\textbf{x}\in \mathbb{R}^{N}$和一个使用$\theta\in \mathbb{R}^{N}$进行参数化的滤波器$\textbf{g}_{\theta}=\text{diag}(\theta)$的乘积：
$$
\textbf{g}_{\theta}\star \textbf{x}=\textbf{Ug}_{\theta}(\Lambda)\textbf{U}^T\textbf{x}, \qquad (5.1)
$$
其中$U$是归一化后的图Laplacian矩阵$L=I_N-D^{- \frac{1}{2}}AD^{- \frac{1}{2}}=U\Lambda U^T$（$D$是度矩阵，$A$是邻接矩阵）的特征向量矩阵，$\Lambda$是$L$的特征值矩阵（对角）。

这个操作会导致巨大的计算量和非均匀(non-spatially)局部滤波器。

### 5.1.2 Chebnet

Hammond et al.提出$g_{\theta}(\Lambda)$可以通过部分展开到$K$阶的的切比雪夫多项式$\textbf{T}_k(x)$来近似。因此得到操作：
$$
\textbf{g}_{\theta}\star \textbf{x}= \sum\limits_{k=0}^K\theta_k\textbf{T}_k(\tilde{\textbf{L}})\textbf{x}  \qquad (5.2)
$$
其中$\tilde{\textbf{L}}=\frac{2}{\lambda_{\max}}\textbf{L}-\textbf{I}_N$. $\lambda_{\max}$代表$\textbf{L}$中最大的特征值。$\theta\in \mathbb{R}^K$现在是切比雪夫系数向量。切比雪夫多项式的定义为$\textbf{T}_k(\textbf{x})=2\textbf{xT}_{k-1}(\textbf{x})-\textbf{T}_{k-2}(\textbf{x}), \quad \textbf{T}_0(\textbf{x})=1\quad \text{and} \quad \textbf{T}_1(\textbf{x})=\textbf{x}$. 可以观察到由于它是Laplacian的K阶多项式，所以这个操作是K-localized的。

Chebnet使用这个K-localized卷积来定义一个卷积神经网络，不需要计算Laplacian矩阵的特征向量。

### 5.1.3 GCN

Kipf and Welling将逐层卷积操作的K限制为1来缓解节点度分布较为宽广的图上出现的局部近邻结构过拟合问题。进一步近似$\lambda_{\max}\approx 2$，等式简化为：
$$
\textbf{g}_{\theta'}\star \textbf{x} \approx \theta_0'\textbf{x} + \theta_1' (\textbf{L}-\textbf{I}_N)\textbf{x} = \theta_0'\textbf{x} - \theta_1' \textbf{D}^{- \frac{1}{2}}\textbf{A}\textbf{D}^{- \frac{1}{2}}  \qquad (5.3)
$$
有两个自由参数$\theta_0'$和$\theta_1'$. 将参数限制为$\theta=\theta_0'=-\theta_1'$后，可以得到：
$$
\textbf{g}_{\theta}\star \textbf{x} \approx \theta \big(\textbf{I}_N + \textbf{D}^{- \frac{1}{2}}\textbf{A}\textbf{D}^{- \frac{1}{2}} \big)\textbf{x}. \qquad (5.4)
$$
注意到如果重复（堆叠）这一操作会导致数值不稳定和梯度爆炸/消失，作者引入了*renormalization trick:* $I_N+D^{- \frac{1}{2}}AD^{- \frac{1}{2}} \rarr \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$, 其中$\tilde{A}=A+I_N$,$\tilde{D}_{ii}=\sum_j\tilde{A}_{ij}$.最后作者将这一定义推广到了有$C$个输入通道的信号$\textbf{X}\in \mathbb{R}^{N\times C}$和$F$个滤波器的特征图：
$$
\textbf{Z}=\tilde{\textbf{D}}^{-\frac{1}{2}}\tilde{\textbf{A}}\tilde{\textbf{D}}^{-\frac{1}{2}}\textbf{X}\Theta \qquad (5.5)
$$
其中$\Theta\in \mathbb{R}^{C\times F}$是一个滤波器参数的矩阵，$\textbf{Z}\in \mathbb{R}^{N\times F}$是卷积后的信号矩阵。

### 5.1.4 AGCN

上述所有模型都是用原始图结构来表示节点间的关系。但是不同节点间可能存在隐含的关系，因此提出自适应图卷积网络(AGCN)来学习隐含的关系。AGCN学习一个“残差”图Laplacian矩阵$\textbf{L}_{res}$并把它加到原始的Laplacian矩阵上：
$$
\widehat{\textbf{L}} = \textbf{L} + \alpha \textbf{L}_{res} \qquad (5.6)
$$
其中$\alpha$是一个参数。

$\textbf{L}_{res}$通过一个学习的图邻接矩阵$\widehat{\textbf{A}}$计算得到
$$
\textbf{L}_{res}=\textbf{I} - \widehat{\textbf{D}}^{-\frac{1}{2}}\widehat{\textbf{A}}\widehat{\textbf{D}}^{-\frac{1}{2}} \\
\widehat{\textbf{D}} = \text{degree}(\widehat{\textbf{A}})  \qquad (5.7)
$$
$\widehat{\textbf{A}}$通过一个学习的度量计算得到。自适应度量背后的想法是 对于图结构数据来说，欧氏距离度量并不适用，而所采用的度量应该能够根据任务和输入特征自适应调整。AGCN使用归一化的马氏距离：
$$
D(\text{x}_i,\text{x}_j)=\sqrt{(\text{x}_i - \text{x}_j)^T\text{M} (\text{x}_i - \text{x}_j)}, \qquad (5.8)
$$
其中$\text{M}$是一个学习到的参数，满足$\text{M}=\text{W}_d\text{W}_d^T$. 而$\text{W}_d$是自适应空间的变换基。AGCN计算高斯核并将$G$归一化来获得稠密的邻接矩阵$\widehat{\textbf{A}}$:
$$
G_{x_i,x_j} = \exp{(-D(\text{x}_i, \text{x}_j)/(2\sigma^2))}. \qquad (5.9)
$$

## 5.2 Spatial Methods

上面提到的所有谱方法中，学习到的滤波器都依赖于Laplacian特征基，而特征基依赖于图结构。意味着在一个特定图上训练的模型不能直接用于另一个不同结构的图上。

作为对比，稀疏方法直接在图上定义卷积，在稀疏的近邻上进行操作。稀疏方法的主要挑战是在有不同尺寸邻居和保持CNNs的局部不变性的前提下定义卷积操作。

### 5.2.1 Neural FPS

Duvenaud et al. 对度不同的节点使用不同的权重矩阵，
$$
\text{x}=\textbf{h}_v^{t-1}+\sum\limits_{i=1}^{|N_v|}\textbf{h}_i^{t-1} \\
\textbf{h}_v^t=\sigma\Big(\text{x}\textbf{W}_t^{|N_v|} \Big), \qquad (5.10)
$$
其中$\textbf{W}_t^{|N_v|}$是第$t$层中度为$|N_v|$的节点的权重矩阵，$N_v$代表节点$v$的邻居的集合，$\textbf{h}_v^t$是第$t$层中节点$v$的embedding. 从等式中可以看到模型首先将节点自身和其邻居的embeddings加到一起，然后使用$\textbf{W}_t^{|N_v|}$做变换。这个模型对于度不同的节点定义了不同的矩阵$\textbf{W}_t^{|N_v|}$. 这一方法的主要缺陷是无法应用于节点度更高的大规模图上。

### 5.2.2 PATCHY-SAN

PATCHY-SAN模型首先对每个节点选取k个邻居并归一化。然后将归一化后的邻居当作感受野进行卷积操作。具体来说，这个方法有四个步骤：

* **Node Sequence Selection. **这个方法并不处理图中的所有节点，而是选择一个节点序列来处理。首先使用一个给图打标签的流程来获取节点的序然后得到节点序列。然后使用一个步长$s$从序列中选择$w$个节点。
* **Neighborhood Assembly. **这一步中，构建上一步中选择的节点的感受野。每个节点的邻居作为候选，模型使用一个简单的广度优先搜索来收集每个节点的k个邻居。首先提取1阶邻居，然后寻找更高阶邻居直到提取到k个邻居。
* **Graph Normalization. **这一步中，算法的目标是对感受野中的节点给出一个序，所以这一步将无序的图空间映射到一个向量空间。这是最重要的一步，这一步的思想是如果两个不同的图的节点有相似的结构，就将它们分配到相似的相关位置。
* **Convolutional Architecture. **经过上一步对感受野归一化之后，就可以使用CNN架构了。归一化后的邻居作为感受野，节点和边则当作通道。

这个模型的示意见Figure 5.1. 这个方法试图将图学习问题转换成传统的欧几里得数据学习问题。

![Figure 5.1](5.1.png"Figure 5.1")

### 5.2.3 DCNN

Atwood and Towsley提出扩散卷积神经网络(diffusion-convolutional neural networks, DCNNs)。在DCNN中使用变换矩阵来定义节点的邻居。对于节点分类来说，有
$$
\textbf{H}=\sigma (\textbf{W}^c \odot \textbf{P}^*\textbf{X}), \qquad (5.11)
$$
其中$\textbf{X}$是一个$N\times F$的输入特征张量（$N$是节点数，$F$是特征数量）。$\textbf{P}^*$是一个$N\times K\times N$的张量，包含矩阵$\textbf{P}$的幂级数$\{\textbf{P},\textbf{P}^2,...,\textbf{P}^K\}$. $\textbf{P}$是图邻接矩阵$\textbf{A}$经过度归一化后得到的变换矩阵。每个实体都被转换成一个扩散卷积表达，即一个$K\times F$的矩阵，是$F$个特征上的$K$阶扩散。然后通过一个$K\times F$的权重矩阵和一个非线性激活函数$\sigma$进行定义。最终，$\textbf{H}$（$N\times K\times N$）代表图中每个点的扩散表达。

对于图分类问题，DCNN只用节点表达的平均，
$$
\textbf{H}=\sigma(\textbf{W}^c \odot 1_N^T \textbf{P}^*\textbf{X}/N) \qquad (5.12)
$$
这里的$1_N$是一个全1的$N\times 1$的向量。DCNN同样可以用于变得分类任务，需要将边转换成节点并扩充邻接矩阵。

### 5.2.4 DGCN

Zhuang and Ma提出双图卷积网络(dual graph convolutional network, DGCN)共同考虑图的局部一致性和全局一致性。它使用两个卷积网络来捕捉局部/全局一致性，并采用一个无监督损失将二者组合到一起。第一个卷积网络和公式（5.5）一致。第二个网络用正点互信息矩阵(positive pointwise mutual information, PPMI)替换了邻接矩阵：
$$
\textbf{H}'=\sigma \Big(\textbf{D}_P^{-\frac{1}{2}} \textbf{X}_P \textbf{D}_P^{-\frac{1}{2}} \textbf{H} \Theta \Big), \qquad (5.13)
$$
其中$\textbf{X}_P$是PPMI矩阵，$\textbf{D}_P$是$\textbf{X}_P$的对角的度矩阵，$\sigma$是一个非线性激活函数。

联合使用两个视角的动机是：（1）公式（5.5）对局部一致性进行建模，代表邻近节点可能拥有相似的标签；（2）公式（5.13）对全局一致性进行建模，假设有相似上下文的节点也有相似的标签。局部一致性卷积和全局一致性卷积分别称为$Conv_A$和$Conv_P$.

作者后来用最终的loss函数将二者组合在一起：
$$
L=L_0(Conv_A)+\lambda(t)L_{reg}(Conv_A, Conv_P). \qquad (5.14)
$$
$\lambda(t)$是调整两个loss函数重要性的动态权重。$L_0(Conv_A)$是给定节点标签的有监督loss函数。如果要预测$c$个不同的标签，$Z^A$代表$Conv_A$的输出矩阵，$\widehat{Z}^A$代表输出$Z^A$经过softmax操作的结果，然后交叉熵损失$L_0(Conv_A)$就可以写成：
$$
L_0(Conv_A)=-\frac{1}{|y_L|}\sum\limits_{l\in y_L}\sum\limits_{i=1}^c Y_{l,i} ln(\widehat{Z}_{l,i}^A), \qquad (5.15)
$$
其中$y_L$是训练集的索引集合，$Y$是gt

$L_{reg}$的计算可以写成：
$$
L_{reg}(Conv_A,Conv_P)=\frac{1}{n}\sum\limits_{i=1}^n \Vert\widehat{Z}_{i,:}^P-\widehat{Z}_{i,:}^A \Vert^2, \qquad (5.16)
$$
其中$\widehat{Z}^P$代表$Conv_P$经过softmax的输出。因此$L_{reg}(Conv_A, Conv_P)$是用来衡量$\widehat{Z}^P$和$\widehat{Z}^A$间差异的无监督loss函数。这个模型的架构如Figure 5.2所示。

![Figure 5.2](5.2.png"Figure 5.2")

### 5.2.5 LGCN

Gao et al.提出了可学习图卷积网络(learnable graph convolutional networks, LGCN)。这个网络基于可学习图卷积层(learnable graph convolutional layer, LGCL)和子图训练策略。

LGCL将CNNs用作聚合器。它在节点的邻居矩阵上使用最大池化来获取top-k特征元素，然后使用1维CNN来计算隐藏表达。LGCL的传播步骤：
$$
\widehat{H}_t=g(H_t,A,k) \\
H_{t+1}=c\Big(\widehat{H}_t \Big), \qquad (5.17)
$$
其中$A$是邻接矩阵，$g(\cdot)$是选择最大的k个节点的操作，$c(\cdot)$代表常规的1维CNN

模型使用选择前k个最大节点的操作来聚合每个节点的信息。对一个给定节点$x$，首先收集（聚合）其邻居的特征；假设它有n个邻居，每个邻居有c个特征，那么就得到了一个矩阵$M\in \mathbb{R}^{n\times c}$. 如果n比k小，则用全零的列补齐。然后对选中的最大k个节点的每一列的值进行排序，并选择top-k个值。然后把节点$x$的embedding插入到这个矩阵的第一行，得到矩阵$\widehat{M}\in \mathbb{R}^{(k+1)\times c}$.

得到矩阵$\widehat{M}$后，使用常规1维CNN将特征聚合。函数$c(\cdot)$应该接收一个$N\times (k+1)\times C$的矩阵作为输入并输出一个$N\times D$或$N\times 1 \times D$的矩阵。Figure 5.3 给出了LGCL的一个例子。

![Figure 5.3](5.3.png"Figure 5.3")

### 5.2.6 MONET

使用$x$代表图中的节点，$y\in N_x$代表节点$x$的邻居节点。MoNet模型计算节点和其邻居间的伪坐标$\textbf{u}(x,y)$，并对这些坐标使用一个赋权函数：
$$
D_j(x)f=\sum\limits_{y\in N_x}w_j(\textbf{u}(x,y))f(y), \qquad(5.18)
$$
其中参数为$\textbf{w}_{\Theta}(\textbf{u})=(w_1(\textbf{u}),...,w_J(\textbf{u}))$，$J$代表提取的patch的尺寸。在非欧域上的卷积的稀疏推广为：
$$
(f\star g)(x)=\sum\limits_{j=1}^J g_jD_j(x)f. \qquad (5.19)
$$
这样，其他的方法就可以被视为对应不同坐标$\textbf{u}$和权重函数$\textbf{w(u)}$的特例。几个设置见Table 5.1

![Table 5.1](t5.1.png"Table 5.1")

### 5.2.7 GraphSAGE

Hamilton et al.提出了GraphSAGE，一个通用的归纳框架。这个框架通过对一个节点的局部邻居的特征进行采样和聚合来生成embeddings. GraphSAGE的传播步骤为：
$$
\textbf{h}_{N_v}^t=\text{AGGREGATE}_t(\{\textbf{h}_u^{t-1},\forall u\in N_v \}) \\
\textbf{h}_v^t=\sigma(\textbf{W}^t \cdot [\textbf{h}_v^{t-1}\Vert \textbf{h}_{N_v}^t]), \qquad (5.20)
$$
其中$\textbf{W}^t$是第$t$层的参数。


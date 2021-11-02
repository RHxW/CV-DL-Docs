# Swin Transformer:Hierarchical Vision Transformer using Shifted Windows
## Abstract
提出一个新的视觉Transformer，称为Swin Transformer，可用作视觉任务的通用骨干网络。为了解决transformer在文字和视觉领域应用上的差异，提出一个分层Transformer，其表达通过切换窗口来计算。切换窗口的方法通过自注意力的计算限制为不重复的局部窗口同时允许交叉窗口连接，实现更高的效率。这一层次架构可灵活适用于不同尺度的模型，并且拥有根据输入尺寸变化的线性计算复杂度。

![Figure 1](1.png 'Figure 1')

## 1. Introduction
神经网络在NLP领域和在CV领域的应用上存在巨大差异，其中就包括尺度。在语言transformers中，词token是最基础的元素，与此不同的是，视觉任务中的基础元素可能根据尺度的变化而不同。现有的基于transformer的模型都将token设定为一个固定的尺度，而这样做其实是不适合与视觉任务的。另一个差异是视觉任务中的分辨率要比文本中词的分辨率大很多。有的任务，比如dense prediction的语义分割，需要像素级别的预测，这对于transformer来说就很难处理。为了解决这些问题，提出一个通用的transformer骨干网络——Swin Transformer，由多层特征图构成，并且根据图片尺寸有线性的计算复杂度。如Figure 1(a)所示，Swin Transformer通过从小尺寸patches（图上灰线）开始并逐渐融合相邻patches，从而构建一个分层的表达。用这些分层的特征图，Swin Transformer可以方便地适用于dense prediction任务。线性计算复杂度是通过在不重合的窗口上计算自注意力实现的。每个窗口中的patches数量是固定的，因此计算复杂度随着图片尺寸线性变化。这些优点使Swin Transformer很适合作为一个通用的backbone应用于多种视觉任务中。
Swin Transformer设计中的一个核心点是连续的自注意力层间窗口划分的切换，如Figure 2所示。窗口的切换连接了上一层中的不同窗口，为窗口间引入了连接，显著提升了模型的建模能力。这一策略在实际应用的延迟上也有效果：同一个窗口内的所有query共享同一个key集合。这种方式在延迟上要优于滑动窗口自注意力机制。

![Figure 2](2.png 'Figure 2')

## 2. Related Work
...

![Figure 3](3.png 'Figure 3')
## 3. Method
### 3.1. Overall Architecture
Swin Transformer的总体架构如Figure 3所示，图中是tiny version(Swin-T). 它首先通过patch分解模块将一个输入图片分解成不重叠的patches，像ViT一样。每一个patch作为一个token处理，其特征设置为将原始的RGB值串联在一起。我们的实现中，使用patch size为
$4\times 4$，因此每个patch的特征尺寸为$4\times 4\times 3=48$. 然后经过一个线性映射层得到一个固定的维度，表示为C
然后这些patch tokens会经过多个有修改后的自注意力计算的Transformer blocks(Swin Transformer blocks). Transformer block会保持token的数量$(\frac{H}{4}\times \frac{W}{4})$，这些Transformer block和线性映射层一起称为Stage 1
为了获取层次表示，随着网络层数加深，通过patch合并层将tokens的数量进行缩减。第一个合并层将每个$2\times 2$范围内相邻patches的特征进行拼接，然后再这个4C维度的拼接特征上使用线性层进行一次映射。这种方式下，token的缩减倍数是$2\times 2=4$（分辨率下降两倍），而输出的维度则设定为2C. Swin Transformer blocks在其后用于特征变换，保持分辨率为$\frac{H}{8}\times \frac{W}{8}$. patch合并和特征变换合称为Stage 2. 将上述操作重复两次，作为Stage 3和Stage 4，对应的输出分辨率分别为$\frac{H}{16}\times \frac{W}{16}$和$\frac{H}{32}\times \frac{W}{32}$. 这些阶段一起作用生成了层次表示，和传统卷积网络的特征图分辨率相同。因此这种结构可以作为现有方法的backbone网络的替换。

**Swin Transformer block** Swin Transformer是通过将Transformer block中标准multi-head self attention(MSA)模块替换为基于切换窗口的模块，而其他部分保持不变来实现的。如Figure 3(b)中所示，一个Swin Transformer block由一个基于切换窗口的MSA模块然后接一个2层的MLP（中间非线性为GELU）构成。每个MSA模块和每层MLP之前都有一个LN层，每个模块之后都有一个残差连接。

### 3.2. Shifted Window based Self-Attention
标准Transformer架构和其在图像分类中的变体都应用了全局自注意力机制，也就是计算每个token和其他所有token间的关系。这种全局的计算导致了$n^2$的计算复杂度，使这种方式与很多视觉任务不匹配。

**Self-attention in non-overlapped windows** 考虑性能，提出在局部窗口内计算自注意力。
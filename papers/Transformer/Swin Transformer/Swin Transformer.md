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

## 3. Method
### 3.1. Overall Architecture

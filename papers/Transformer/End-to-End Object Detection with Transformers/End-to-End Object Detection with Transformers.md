# End-to-End Object Detection with Transformers
## Abstract
提出一个新方法，将目标检测看作一个直接集合预测问题。本方法去掉了传统目标检测流程中的很多人工设计的部分，例如nms或者anchor生成过程。这个新框架的主要部分称为DETR(DEtection TRansformer)，它是一个set-based的全局loss，它强制预测结果经过二分匹配，以及一个transformer 编码解码结构。给定一个关于学习到的物体序列的固定的小集合，DETR推理出物体间以及全局图片上线问间的关系，直接并行输出最终的预测集合。

## 1 Introduction
目标检测的目的是要预测一个bounding boxes的集合以及对应目标的分类。现代检测器采用了一种不直观的方法来解决这一预测任务，比如在大量的proposals、anchors或window centers上定义一个代理回归和分类问题。这种方式的性能受后处理过程的影响比较严重。为了简化这一流程，提出一个直接的集合预测的方法，绕开了各种代理任务。
将物体检测任务视作直接集合预测问题从而精简了训练流程。采用了transformer的encoder-decoder结构，其自注意力机制能够对一个序列中不同元素的内在关系进行建模，使这种结构特别适合集合预测的特殊限制，例如去除重复预测。
提出的DETR同时预测全部目标，并且通过端到端的方式进行训练，loss函数在预测和gt目标间采用二分匹配的方式。DETR抛弃了多个使用先验知识的人工部分，从而简化了检测流程，例如空间anchors和nms操作。和目前大多检测方法不同的是，DETR不需要任何定制层，因此可以在任何包含CNN和transformer类的框架中实现。
我们的匹配loss函数将每个预测单独分配给一个gt目标，并且不随预测目标顺序的变化而变化。
DETR的训练与标准OD检测器有一些区别。新模型的训练时长会特别长。

## 2 Related Work
### 2.1 Set Prediction
检测任务需要避免临近的重复预测，大多数检测器通过使用例如nms等后处理方法来解决这一问题，而直接集合预测则不需要后处理。他们需要能够对全部预测内部建模的全局推理方法来避免冗余。对于固定尺寸的集合预测，可以使用密集全连接网络，但是它太costly.一种通用的方法是使用自动回归序列模型，例如rnn. 但是无论何种方式，loss函数都应该不受预测序列顺序的影响。一般的解决办法是设计一个基于匈牙利算法的loss函数，来找到一个在gt和预测结果间的二分匹配。这种方法保证了顺序不变性，并且每个目标都有唯一的匹配。本方法沿用了这种二分匹配的loss方法。

## 3 The DETR model
在检测领域，有两点是直接集合预测所必须的：
1. 一个集合预测loss用来监督预测box和gt box间一对一的匹配关系
2. 一种可以（在一个单独回路中）预测一组目标并对它们之间关系进行建模的架构

### 3.1 Object detection set prediction loss
在一次传过decoder的过程中，DETR推理得到固定尺寸的N个预测结果集合，其中N被设置为比一般图像中包含的目标数量大很多的数。训练中最主要的困难之一是如何对预测的结果根据对应gt进行打分。我们的loss会给出预测和gt间最优的二分匹配结果，并对box的lossse进行优化。
用y代表检测目标的gt集合，用$\hat{y}=\{\hat{y}_i\}_{i=1}^N$代表N个预测。假设N大于图片中物体的数量，将y视为一个规模为N的集合，剩余的用$\varnothing$补齐（没有物体）。为了找到这两个集合间的二分匹配，需要找到这N个元素$\sigma \in \mathfrak{G}_N$的一个序列，使下述损失最小：
$$
\hat{\sigma}=\argmin_{\sigma \in \mathfrak{G}_N} \sum_{i}^N\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})
$$
其中$\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})$是gt $y_i$和索引为$\sigma(i)$的预测结果间的pair-wise匹配损失。这个最优匹配是使用匈牙利算法计算得到的。
匹配损失同时考虑了类别预测结果和预测与gt间的相似度。gt集合中的每个元素i可以视为$y_i=(c_i,b_i)$，其中$c_i$是目标类别标签（可能是$\varnothing$），$b_i\in[0,1]^4$是一个定义gt box相对于图片尺寸的中心坐标和宽高的向量。定义索引为$\sigma(i)$的预测结果的类别概率为$\hat{p}_{\sigma(i)}(c_i)$, 预测的box为$\hat{b}_{\sigma(i)}$. 所以有：
$$
\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})=-\mathbb{I}_{\{c_i\ne \varnothing\}}\hat{p}_{\sigma(i)}(c_i)+\mathbb{I}_{\{c_i\ne \varnothing\}}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})
$$
这一寻找匹配的过程和现代检测器中匹配gt物体与proposal或者anchor的启发式匹配规则发挥相同的作用。最主要的区别是我们需要找到一对一的匹配关系。
第二部是计算loss函数，为上一步中所有匹配计算匈牙利loss. 定义的loss和一般物体检测其的loss相似，也是由类别预测的负log-likelihood和box loss的线性组合：
$$
\mathcal{L}^{Hungarian}(y,\hat{y})=\sum_{i=1}^N\Big[-\log\hat{p}_{\sigma(i)}(c_i)+\mathbb{I}_{\{c_i\ne \varnothing\}}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)}) \Big]
$$
其中$\hat{\sigma}$是第一步计算的最优匹配。实际上当当前类别对应$c_i=\varnothing$的时候，我们将log-probability权重下调10倍以解决类别不平衡的问题。这与Faster R-CNN中采用的二次采样策略类似。注意到一个物体和$\varnothing$间的匹配损失不依赖于预测结果，意味着这种情况的损失是常数。在计算匹配损失的时候，使用概率而非log-probabilities，这样可以让类别预测项和box项数值上统一，从而达到更好的性能。
**Bounding box loss.** 匹配损失和匈牙利损失的第二部分是给box打分的$\mathcal{L}_{box}(\cdot)$. 和一些根据初始化猜测进行box预测的检测方法不同，本方法直接预测box.常用的l1loss对于小尺寸和大尺寸的box有不同尺度，尽管它们的错误几率差不多。为了缓和这一问题，使用l1 loss和归一化IoU loss的线性组合：
$$
\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})=\lambda_{iou}\mathcal{L}_{iou}(b_i,\hat{b}_{\sigma(i)})+\lambda_{L1}\lVert b_i-\hat{b}_{\sigma(i)}\rVert_1
$$
![Figure 2](2.png 'Figure 2')
### 3.2 DETR architecture
整个DETR的架构如Figure 2所示，特别简单。它包含三个主要部分：一个CNN backbone用来提取特征表达、一个encoder-decoder transformer以及一个简单的前馈网络得到最终检测预测结果。
**Backbone.** 从初始图片$x_{img}\in \mathbb{R}^{3\times H_0\times W_0}$开始，使用一个CNN backbone 生成一个低分辨率特征图$f\in\mathbb{R}^{C\times H\times W}$.使用的值分别为$C=2048,H=\frac{H_0}{32},w=\frac{W_0}{32}$.
**Transformer encoder.** 首先用一个1*1的卷积将f的通道数从C降到d.得到一个新特征图$z_0\in\mathbb{R}^{d\times H\times W}$. 编码器的输入应该是一个序列，因此将$z_0$的空间维度折叠成1维，得到一个尺寸为$d\times HW$的特征图。每个encoder层都有一个标准结构，都由一个多头自注意力模块和一个前馈网络组成。由于transformer架构有顺序不变的性质，因此添加了固定的位置编码。
**Transformer decoder.** decoder采用transformer的标准结构。与原始transformer不同的是，我们的模型并行处理N个物体的解码操作，每个decoder层处理一个。由于decoder也具有顺序不变的性质，因此N个输入embeddings必须是不同的才能获得不同的结果。这些输入的embeddings是我们成为object queries的学习到的为只编码，而且与encoder类似，我们将它们加到每个注意力层的输入上。N个object queries经过decoder的变换得到一个输出embedding.然后各自经过一个前馈网络解码得到box坐标和类别标签，得到N个最终预测结果。通过在这些embeddings上使用自注意力和encoder-decoder注意力，模型能够使用整个图片作为上下文，从全局的角度对所有物体间的关系进行推理。
**Prediction feed-forward networks(FFNs).** 最终的预测通过一个3层带ReLU的感知机（隐藏维度为d）以及一个线性层构成.FFN预测归一化坐标，线性层使用softmax预测类别标签。由于我们预测一个固定数量的bbox，而数量N通常比图片中物体数量要多，所以加入一个空标签$\varnothing$. 这个类别和标准检测方法使用的背景类差不多。
**Auxiliary decoding losses.** 
# Partial FC
## Abstract

## Introduction
contributions：
1. 提出一个softmax的近似算法，可以只使用10%的类数据而保证精度不降低
2. 提出一个分布式训练策略
3. Glint360k数据集

## Method
### Problem Formulation
**Model parallel** 在使用大量id进行人脸识别模型训练的时候，受单张显卡显存的限制，如果不使用模型并行那么训练过程会很痛苦。瓶颈在于如何储存softmax的权重矩阵$W\in \mathbb{R}^{d\times C}$，其中d代表特征向量维度，C代表类别数量。一个简单又直观的的解决办法是将W分成k个子矩阵w，每个的尺寸是$d\times \frac{C}{k}$，然后放在不同的GPU上。所以如果要计算最终的softmax输出，就需要从所有的GPU上收集特征。这样的softmax函数的定义是：
$$
\sigma (X,i)=\frac{e^{w_i^TX}}{\sum^C_{j=1}e^{w_j^TX}}
$$
分子的计算可以由各个GPU独立完成
但是要计算分母，就需要从所有GPU收集信息。一般来说，可以先计算每个GPU自身的值，然后通过通信方式计算全局值。与原始的数据并行相比，这种实现方式的通讯损耗可以忽略不计。数据并行的方式需要将整个W的梯度进行传输从而更新全部权重，而模型并行值传输本地值，而这种损耗是可以忽略的。我们使用集合通信和矩阵操作来描述模型并行的计算过程，如Algorithm 1所示。这种方法可以大幅度降低worker间的通讯。$W, x_i, \nabla X$的尺寸分别为$d*C,N*d,N*d*k$，对于大规模的分类任务有$C \gg N*(k+1)$，其中N代表每个GPU上的mini-batch
![Algorithm 1](a1.png 'Algorithm 1')
**Memory Limits Of Model Parallel** 模型并行的方法可以完全解决w在存储和通信上的问题，因为无论C多大，都可以通过增加更多GPU来解决。所以每个GPU上存储的子矩阵w是不变的，也就是：
$$
Mem_{w}=d\times \frac{C\uparrow}{k\uparrow}\times 4 bytes.
$$
但是w并不是唯一存储在GPU显存中的数据。预测的logits结果的存储也受batch-size的增长影响。将每个GPU上存储的logits表示为$logits=Xw$，那么每个GPU上logits的存储消耗为：
$$
Mem_{logits}=Nk\times \frac{C}{k}\times 4 bytes
$$
其中N是每个GPU的mini-batch大小，k是GPU的个数。假设每个GPU的batch-size是常数，当C增大的时候，为了保持$\frac{C}{k}$不变，就要同时增大k.因此被logits占用的显存会继续增加，因为特征的batch-size随着k的增大同步增加。假设只考虑分类层，每个参数占用12字节。如果使用CosFace或者ArcFace算法，则logits中每个元素占用8字节。因此分类层的显存占用为：
$$
Mem_{FC}=3\times Mem_W+2\times Mem_{logits}
$$
如Figure 2所示，假设每个GPU的batch-size是64，特征向量维度512，那么100万的分类任务需要8块GPU，而1000万的任务就需要至少80块GPU。我们发现logits占用的空间是w的十倍，就使logits的存储成为了模型并行的新瓶颈。结论就是，大量id的分类训练不能通过简单地增加GPU数量来解决

### Approximate Strategy
**Roles of positive and negative classes** 经过对softmax公式的分析，得到了如下假设。如果想要选择W的一个子集来近似softmax，那么必须选中正类中心，而负类中心只需要选择一部分，同时保证模型的性能不下降。

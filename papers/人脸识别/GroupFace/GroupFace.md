# GroupFace

## 1. Introduction
分组，是高效且灵活地生成非常大量人员表示并简要描述一个未知人员的核心思想。每个人脸上都有属于他的特点。同时，常见的特点会出现在一群人的描述中。实际上，包含常见特征的基于分组的描述虽然不能直接定位到个人，但仍然可以帮助缩小搜索范围。可问题是，明确的分组需要手动对巨量的数据进行标注，而且可能受人类感知的有限范围的描述所限。尽管如此，采用分组的概念，可以使识别网络缩小搜索空间并灵活地生成大量id的特征表示。

## 3. Proposed Method
通过使用一个自行分布的分组方法学习潜在分组，构建多个分组感知表示并将它们组合成基于实例的标准表示来增强人脸识别特征的表达能力。
### 3.1. GroupFace
**Instance-based Representation.** 将传统人脸识别方法中的特征向量称为*基于实例的表达*。一般来说，基于实例的表达是通过softmax类loss方法进行训练得到的，并且获得身份的预测：
$$
p(y_i|x)=\text{softmax}_k(g(v_x))
$$
其中$y_i$是身份标签，$v_x$是给定样本$x$的基于实例的表达，$g$是一个将512维特征向量映射到M维空间的函数，M是id数量。
**Group-aware Representation.** GroupFace使用一个新的分组感知表达和基于实例的表达一起增强特征。每个分组感知表达向量都通过对应分组的全连接层提取得到。GroupFace的embedding 特征($\bar{\textbf{v}}_X$，Figure 2中的最终表达)是通过对基于实例的表达$\textbf{v}_X$和分组感知表达$\textbf{v}_X^G$的加权和进行聚合操作得到的。GroupFace使用最终的增强后的表达来预测id信息：
$$
\begin{align*}
p(y_i|x)&=\text{softmax}_k(g(\bar{\textbf{v}}_X)) \\
&=\text{softmax}_k(g(\textbf{v}_X+\textbf{v}_X^G))
\end{align*}
$$
**Structure.** 
# Learning to Cluster Faces on an Affinity Graph

## 1. Introduction

要利用无标签数据的一个自然的想法就是将他们聚类成“伪类别”，这样就可以把这些数据当作有标签数据使用，传入有监督学习流程中。这些聚类一般采取无监督方式，如K-means，spectral clustering，hierarchical clustering和approximate rank-order，将无标签的人脸分组。这些方法依赖于简单的假设，例如K-means隐式地假设每一个族的样本都围绕一个中心；spectral clustering需要族的大小相对平衡等。所以这些方法缺少处理复杂聚类结构的能力，因此常会得到带噪音的聚类结果，特别是在真实世界大尺寸数据集上使用的时候。这个问题严重地限制了这种方法的性能提升。


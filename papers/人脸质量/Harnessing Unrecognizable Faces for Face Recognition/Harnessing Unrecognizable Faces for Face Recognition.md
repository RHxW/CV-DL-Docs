# Harnessing Unrecognizable Faces for Face Recognition
## Abstract
提出一种用于衡量人脸图片可被识别能力的方法，利用了基于经验的观察结论：使用绝大多数可识别样本进行训练的深度模型对人脸图片提取的特征，会在超球面上产生分裂，令识别能力低的图片聚到一起。
因此利用与这个“无法识别id”的距离作为识别能力的度量。
![Figure 1](1.png 'Figure 1')
## 1. Introduction
有理由相信UIs(unrecognizable identity)的特征在高维空间中沿决策边界分布。我们观察到这样的现象：当FR系统使用的训练集中不包含UIs的时候，如果使用该特征进行聚类，则UIs会在特征空间聚集到一起。该现象如Figure 1所示。

## 2. Face recognizability in face recognition

### 2.2. Accounting for Recognizability
假设与UI聚类中心的距离可以作为人脸识别能力的度量，基于此，使用这个距离作为ERS(embedding recognizability score), 不需要额外的训练也不需要标注。
#### 2.2.1 The Embedding Recognizability Score(ERS)
将ERS定义为向量与UI平均向量间的距离。用WIDERFace数据集获取大量的UI图像。在数据集上运行聚类算法之后，结果中最大的聚类只包含UIs. 然后用该聚类归一化后的平均特征$f_{UI}$作为UI特征。给定一个特征向量$f_i$，其对应的ERS$e_i$为：
$$
e_i=\min(1-\langle f_{UI},f_i\rangle,1)
$$
在Figure 5中展示了ERS和识别能力间的关系。
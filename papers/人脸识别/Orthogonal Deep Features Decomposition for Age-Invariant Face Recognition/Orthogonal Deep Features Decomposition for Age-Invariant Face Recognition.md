# Orthogonal Deep Features Decomposition for Age-Invariant Face Recognition

## Abstract
为了减小由年龄变化引起的类内差异，本文提出一个新的用于学习年龄不变人脸特征的方法。具体来说，将人脸特征分解成相互正交的两部分，分别代表年龄相关特征和身份相关特征。然后用对于年龄变化鲁棒的身份特征来进行人脸识别任务。

## 1 Introduction
AIFR(age-invariant face recognition)的主要挑战如Figure 1所示。当前AIFR主要有两个方向：生成法和判别法。生成法有两个问题：1）将识别流程分成了两部分，无法实现端到端识别，效率较低。2）生成模型的不稳定性会影响识别效果。
本作的目标是设计一个新的深度学习方法来高效地学习从混合了年龄信息的特征中提取年龄无关的部分。其中心思想是将人脸特征分解成与年龄相关以及与身份相关的两部分，其中与身份相关的部分不受年龄变化影响，适用于AIFR任务。

## 2 Proposed Approach
### 2.1 Orthogonal Deep Features Decomposition
AIFR领域有两个主要困难，分别是不同年龄间身份信息的巨大差异，和通用框架提取特征的信息混合。混合的特征从本质上讲降低了跨年龄人脸识别的鲁棒性。为了解决这一问题，提出一个新的方法称为正交嵌入CNN(orthogonal embedding CNNs).
给定一个FC层输出特征x，将其分解成两部分。一部分$x_{id}$与身份相关，另一部分$x_{age}$与年龄相关。这样，如果从x中将$x_{age}$移除，就可以获得与年龄无关的身份特征$x_{id}$.本文提出一个新的方法以正交的方式对$x_{age}$和$x_{id}$进行建模。
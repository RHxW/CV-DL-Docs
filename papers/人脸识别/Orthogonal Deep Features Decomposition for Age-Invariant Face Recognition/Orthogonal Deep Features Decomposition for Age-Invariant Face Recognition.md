# Orthogonal Deep Features Decomposition for Age-Invariant Face Recognition

## Abstract
为了减小由年龄变化引起的类内差异，本文提出一个新的用于学习年龄不变人脸特征的方法。具体来说，将人脸特征分解成相互正交的两部分，分别代表年龄相关特征和身份相关特征。然后用对于年龄变化鲁棒的身份特征来进行人脸识别任务。

## 1 Introduction
AIFR(age-invariant face recognition)的主要挑战如Figure 1所示。当前AIFR主要有两个方向：生成法和判别法。生成法有两个问题：1）将识别流程分成了两部分，无法实现端到端识别，效率较低。2）生成模型的不稳定性会影响识别效果。
本作的目标是设计一个新的深度学习方法来高效地学习从混合了年龄信息的特征中提取年龄无关的部分。其中心思想是将人脸特征分解成与年龄相关以及与身份相关的两部分，其中与身份相关的部分不受年龄变化影响，适用于AIFR任务。

## 2 Proposed Approach
### 2.1 Orthogonal Deep Features Decomposition
AIFR领域有两个主要困难，分别是不同年龄间身份信息的巨大差异，和通用框架提取特征的信息混合。混合的特征从本质上讲降低了跨年龄人脸识别的鲁棒性。为了解决这一问题，提出一个新的方法称为正交嵌入CNN(orthogonal embedding CNNs).
给定一个FC层输出特征x，将其分解成两部分。一部分$x_{id}$与身份相关，另一部分$x_{age}$与年龄相关。这样，如果从x中将$x_{age}$移除，就可以获得与年龄无关的身份特征$x_{id}$.本文提出一个新的方法以正交的方式对$x_{age}$和$x_{id}$进行建模。在A-Softmax方法中，不同id的数据在角度上进行分离，受此启发，我们将特征x在球坐标系中进行分解$x_{sphere}=\{r;\phi_1,\phi_2,...,\phi_n\}$.角度部分$\{\phi_1,\phi_2,...,\phi_n\}$代表身份相关的信息，半径r则用于编码年龄信息。一般来说，$x\in R^n$在$x_{sphere}$下分解为
$$
x=x_{age}\cdot x_{id}
$$
其中$x_{age}=\lVert x \rVert_2, x_{id}=\{\frac{x_1}{\lVert x \rVert_2},\frac{x_2}{\lVert x \rVert_2},...,\frac{x_n}{\lVert x \rVert_2}\},\lVert x_{id} \rVert_2=1$，方便起见，用$n_x$代表$\lVert x \rVert_2$，用$\tilde{x}$代表$\frac{x}{\lVert x \rVert_2}$
![Figure 2](2.png 'Figure 2')
### 2.2 Multi-Task Learning
最后一个FC层输出的特征x会被分解成$x_{age}$和$x_{id}$. 提出的CNN模型的架构如Figure 2所示。
**Learning age-related component.** 为了挖掘年龄信息的线索，利用一个年龄估计的任务来学习$x_{age}(n_x)$与真实年龄之间的关系。简单起见，采用线性回归做年龄估计任务，其loss为：
$$
L_{age}=\frac{1}{2M}\sum_{i=1}^M \lVert f(n_{x_i})-z_i \rVert^2_2
$$
其中$n_{x_i}$是第i个特征$x_i$的L2范数，$z_i$是对应的第i个年龄标签。$f(x)$是一个用于将$n_{x_i}$和$z_i$联系起来的映射函数。实际使用的是一个线性变换。
**Learning identity-related component.** 人脸识别系统在实际使用的时候，只会用到归一化特征$\tilde{x}$. 因此与身份相关的部分$x_{id}$的判别能力应该尽可能地强。使用一个与A-Softmax类似的loss函数：
$$

$$
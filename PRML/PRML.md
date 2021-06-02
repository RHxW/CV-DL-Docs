# Pattern Recognition and Machine Learning
****

# Chapter 1. Introduction
## 1.1
使用正则化方法减轻过拟合现象，在误差函数中引入一个正则化项以阻止权重参数值变很大，如：
$$
\tilde{E}(\textbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2+\frac{\lambda}{2}\lVert \textbf{w} \lVert^2  \qquad (1.4)
$$
其中w是权重向量($\lVert \textbf{w} \lVert^2 \equiv \textbf{w}^T \textbf{w} = w_0^2+w_1^2+...+w_M^2$)，$\lambda$是用来调整正则化项与其他部分比重的参数。
这个误差函数可以在闭式解下最小化。
这种技术在统计学上称为*收缩*方法(*shrinkage* methods)，因为它缩小了参数的值。
平方正则化方法称为*岭回归*(*ridge regression*)。在神经网络中这种方法称为*权重衰减*(*weight decay*)。

实际上$\lambda$控制了模型的有效复杂度，因此也决定了过拟合的程度。


## 1.2 Probability Theory
条件概率: $P(A|B)=\frac{P(AB)}{P(B)}$
有: $P(AB)=P(A|B)P(B)$
由对称性: $P(AB)=P(BA)$
得: $P(A|B)P(B)=P(B|A)P(A)$
整理得*Bayes' theorem*: $P(A|B)=\frac{P(B|A)P(A)}{P(B)}$
以及: 
$$
p(X)=\sum_{Y}p(X|Y)p(Y)
$$

### 1.2.1 Probability densities
对连续变量$x$，其概率密度$p(x)$，则在区间$(a,b)$上的概率为：
$$
p(x\in (a,b))=\int_a^bp(x)\text{d}x
$$
性质：
$$
p(x) \ge 0 \\
\int_{-\infin}^{\infty}p(x)\text{d}x=1. \\
P(z)=\int_{-\infty}^zp(x)\text{d}x \\
P'(x)=p(x)
$$

### 1.2.2 Expectations and covariances
一个函数$f(x)$在一个概率分布$p(x)$下的平均值称为$f(x)$的*期望(expectation)*，表示为$\mathbb{E}[f]$，离散和连续形式分别为：
$$
\mathbb{E}[f]=\sum_x p(x)f(x) \\
\mathbb{E}[f]=\int p(x)f(x) \text{d}x.
$$
也就是说，平均值是通过不同$x$的取值的概率进行加权得到的。

多变量的函数，用下标表示对哪个变量进行平均：
$$
\mathbb{E}_x[f(x,y)]
$$
代表函数$f(x,y)$关于x分布的平均。注意它是一个关于$y$的函数。

在条件分布下的条件期望：
$$
\mathbb{E}_x[f|y]=\sum_x p(x|y)f(x)
$$

方差var
$$
\text{var}[f]=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]
$$
方差代表函数$f(x)$与其期望$\mathbb{E}[f(x)]$间有多大的差异。
上式展开得：
$$
\text{var}[f]=\mathbb{E}[f(x)^2]-\mathbb{E}[f(x)]^2
$$
对于x：
$$
\text{var}[x]=\mathbb{E}[x^2]-\mathbb{E}[x]^2
$$

两个随机变量的协方差：
$$
\text{cov}[x,y]=\mathbb{E}_{x,y}[\{x-\mathbb{E}[x]\}\{y-\mathbb{E}[y]\}] \\
\quad = \mathbb{E}_{x,y}[xy]-\mathbb{E}[x]\mathbb{E}[y]
$$

表示x和y一起变动的范围，如果x和y相互独立，则协方差为0

### 1.2.3 Bayesian probabilities
1. 概率的*经典*或*频率学派*视角：随机、可重复事件的频率
2. *贝叶斯*视角：概率对不确定性进行了量化

对于多项式回归中的参数$\textbf{w}$，在观察数据之前，有先验概率$p(\textbf{w})$，观察到的数据所造成的影响$\mathcal{D}=\{t_1,...,t_N\}$表示为条件概率$p(\mathcal{D}|\textbf{w})$,那么贝叶斯的形式为：
$$
p(\textbf{w}|\mathcal{D})=\frac{p(\mathcal{D}|\textbf{w})p(\textbf{w})}{p(\mathcal{D})}
$$
这允许我们在得到对$\mathcal{D}$的观察后，以后验概率$p(\textbf{w}|\mathcal{D})$的形式来评估$\textbf{w}$的不确定性。
其中等式右边的$p(\mathcal{D}|\textbf{w})$对观测数据进行评估，可视为一个关于参数向量$\textbf{w}$的函数，它被称为*似然函数(likelihood function)*. 它表示对于不同的$\textbf{w}$，得到观察数据的概率。似然并不是概率分布，它的积分也自然不等于1.
有了似然的定义，就可以将Bayes' theorem用语言来描述：
$$
\text{posterior} \propto \text{likelihood} \times \text{prior}
$$
其中所有量都视作$\textbf{w}$的函数。

无论对于贝叶斯还是频率范式，似然函数$p(\mathcal{D}|\textbf{w})$的地位都很重要。但是它们的用法却有这根本性的不同。在频率方法中，将$\textbf{w}$视作固定的参数，它的值取决于某种'estimator'，误差是通过不同的可能数据集$\mathcal{D}$的分布得到的。
而对于贝叶斯观点，只有一个数据集$\mathcal{D}$（也就是真正观察到的那个），而参数的不确定性是通过$\textbf{w}$的概率分布来反映的。

经常使用的一种频率估计是*最大似然(maximum likelihood)*，这种方法将$\textbf{w}$设定为使似然函数$p(\mathcal{D}|\textbf{w})$最大的值。

贝叶斯观点的一大优势在于对先验知识的包含是自然而然的。举例来说，假设抛一个fair-looking的硬币三次，结果全是正面朝上。一个典型的最大似然估计是正面朝上的概率是1，意味着未来抛硬币也都会是正面朝上。而任意先验的贝叶斯方法都会得出没这么极端的结论。


### 1.2.4 The Gaussian distribution

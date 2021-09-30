# Pattern Recognition and Machine Learning
****
[toc]
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


### 1.2.5 Curve fitting re-visited


### 1.2.6 Bayesian curve fitting

## 1.3. Model Selection

我们已经知道在最大似然方法中，由于过拟合现象的存在，模型在训练集上的表现并不能很好地代表它在未见过数据集上的表现。如果数据量很大的话，那么一个简单的方法是用其中的一部分来训练得到一批模型，或者得到一个模型的复杂度参数的取值范围，然后用称为*验证集*的独立数据对模型进行比较，选择一个性能最好的。如果模型在有限的数据集上迭代了很多次，那么有可能在验证集上出现过拟合现象，这就有必要保留第三个集合：*测试集*用来最终评估模型的性能。

历史上提出了多个‘信息准则’，尝试通过增加一个用于补偿复杂模型的过拟合现象的惩罚项来修正最大似然的偏置。例如，*赤池信息量准则(Akaike information criterion, AIC)*，选择令：
$$
\ln{p(\mathcal{D}|\textbf{w}_{ML})}-M
$$
最大的模型。其中$p(\mathcal{D}|\textbf{w}_{ML})$是最佳拟合对数似然，$M$是模型中可调参数的数量。
这种准则没有将模型参数的不确定性纳入其考虑中，实际上它们倾向于选择过于简单的模型。

## 1.4. The Curse of Dimensionality
我们对几何的直观理解是在三维空间的一生所形成的，而这种直觉在更高维空间就一点用处都没有了。举一个简单的例子，考虑$D$维空间中一个半径$r=1$的超球面，求$r=1-\epsilon$和$r=1$之间部分的体积分数。将$D$维空间中半径为$r$的超球面的体积表示为：
$$
V_D(r)=K_D r^D
$$
其中常数$K_D$依赖于维度$D$，那么要求的分数就可以表示为：
$$
\frac{V_D(1)-V_D(1-\epsilon)}{V_D(1)}=1-(1-\epsilon)^D
$$
可以看到对于较大的$D$，尽管$\epsilon$很小，这个分数也接近于1. 也就是说，在高维空间中，一个超球面的绝大部分体积都集中在其表面附近的薄壳中！

考虑高维空间中高斯分布的性质。如果从笛卡尔坐标系转向极坐标系，然后整合方向变量，就得到与原始函数半径为$r$的函数密度$p(r)$的表示。因此$p(r)\delta r$就是半径$r$处厚度为$\delta r$的薄壳所包含的概率质量。

在高维空间中可能会出现的严重的困难有时称为*维度诅咒(curse of dimensionality)*
尽管维度诅咒会在模式识别应用中引发重要的问题，我们仍可以找到应用于高维空间的有效的技术。原因有二：
1. 真实数据一般会分布于一个效维度比较低的小范围空间中
2. 真实数据基本上都会有一些平滑性质（至少局部有平滑性质），也就意味着输入变量发生较小变化会导致输出变量发生较小变化，那么就可以利用类似局部插值的方法来进行预测

## 1.5. Decision Theory

### 1.5.1 Minimizing the misclassification rate
$\mathcal{C}_k$是类别$\mathcal{R}_k$是*决策区域(decision regions)*，每个类别对应它的决策区域。决策区域间的边界称为*决策边界(decision boundaries)*或*决策界面(decision surfaces)*. 决策区域不必须是连续的，也可由多个不相交的区域构成。
以两个类别为例，当把属于$\mathcal{C}_1$的样本分配给$\mathcal{C}_2$或相反的时候，就发生了错误，错误发生的概率为：
$$
p(\text{mistake})=p(\textbf{x}\in \mathcal{R}_1, \mathcal{C}_2)+p(\textbf{x}\in \mathcal{R}_2, \mathcal{C}_1) \\
=\int_{\mathcal{R}_1}p(\textbf{x}, \mathcal{C}_2)\text{d}\textbf{x}+\int_{\mathcal{R}_2}p(\textbf{x}, \mathcal{C}_1)\text{d}\textbf{x}
$$

显然应该把$\textbf{x}$指定为令上式更小的类别。也就是说，当$\textbf{x}$的每一个取值指定的类别的后验概率$p(\mathcal{C}_k|\textbf{x})$最大的时候，错误概率最小。

### 1.5.2 Minimizing the expected loss

### 1.5.3 The reject option
控制置信度阈值$\theta$

### 1.5.4 Inference and decision
我们将分类问题分解成了两个阶段，*推理阶段(inference stage)*中使用训练数据对$p(\mathcal{C}_k|\textbf{x})$学习一个模型，随后的*决策阶段(decision stage)*使用这些后验概率做最优的类别分配。一个可替代的方案是将两个问题整合成一个问题，只学习一个函数用来将输入$\textbf{x}$直接映射到决策，这样的函数称为*判别函数(discriminant function)*.
实际上有三种不同的方法，按复杂度降低的顺序：
1. 首先解决用于决定每个类别$\mathcal{C}_k$的类别条件密度$p(\textbf{x}|\mathcal{C}_k)$的推理问题。同时得出先验类别概率$p(\mathcal{C}_k)$. 然后使用Bayes' theorem得到后验类别概率$p(\mathcal{C}_k|\textbf{x})$. 然后利用决策理论决定每个新样本的类别。显示或隐式对输入或输出的分布进行建模的方法称为*生成式模型(generative models)*，因为通过在分布中进行采样可以生成人造的数据。
2. 首先得到后验概率$p(\mathcal{C}_k|\textbf{x})$，然后利用决策理论决定每个新样本的类别。直接对后验概率进行建模的模型称为*判别式模型(discriminative models)*.
3. 找到一个函数$f(\textbf{x})$，称为判别函数，它将每个输入$\textbf{x}$直接映射到一个类别标签。这种方法没用到概率。

使用后验概率$p(\mathcal{C}_k|\textbf{x})$的有力理由包括：
**Minimizing risk.**
**Reject option.**
**Compensating for class priors.**
**Combining models.**

### 1.5.5 Loss functions for regression
loss函数$L(t,y(\textbf{x}))$，期望：
$$
\mathbb{E}[L]=\int \int L(t,y(\textbf{x}))p(\textbf{x},t) \text{d}\textbf{x}\text{d}t.
$$
目标是通过选择$y(\textbf{x})$来最小化$\mathbb{E}[L]$.
使用变分法可得：
$$
\frac{\delta\mathbb{E}[L]}{\delta y(\textbf{x})}=
2\int \{y(\textbf{x})-t\}p(\textbf{x},t)\text{d}t=0.
$$

与分类相似，同样有三种方法解决回归问题，按复杂度降序：
1. 首先解决推理问题得到联合密度$p(\textbf{x},t)$. 然后归一化得到条件密度$p(t|\textbf{x})$, 最终边缘化得到条件均值
2. 首先解决推理问题得到条件密度$p(t|\textbf{x})$, 然后边缘化得到条件均值
3. 通过训练数据直接得到一个回归函数$y(\textbf{x})$


## 1.6. Information Theory
考虑离散随机变量$x$, 我们想要知道当观察到这个随机变量的一个具体值的时候获取的信息有多少。信息的量可以看作学习$x$值的‘惊讶度(degree of surprise)’。
信息量的衡量方法基于概率分布$p(x)$, 因此要寻找一个根据$p(x)$单调的量$h(x)$来表示信息量。注意$h(\cdot)$的形式，假如有两个不相关的事件$x,y$, 那么同时观察二者得到的信息量和分别观察它们得到的信息量应该是相同的，也就是说$h(x+y)=h(x)+h(y)$. 而两个不相关的事件在统计上是相互独立的，因此$p(x,y)=p(x)p(y)$. 那么可得：
$$
h(x)=-\log_2p(x)
$$
概率低的事件有更高的信息量。对数底用的是2，因为$h(x)$的单位是比特('binary digits')

平均信息量称为熵，是上式的期望：
$$
\text{H}[x]=-\sum_xp(x)\log_2p(x)
$$
有$\lim_{p\rightarrow 0}p\ln p$

非均匀分布的熵比均匀分布要小一些

### 1.6.1 Relative entropy and mutual information
考虑未知分布$p(\textbf{x})$, 假设使用一个近似分布$q(\textbf{x})$对它进行建模。如果使用$q(\textbf{x})$来构建一个转换$\textbf{x}$的编码方案，则使用$q(\textbf{x})$而非真实分布$p(\textbf{x})$来表示$\textbf{x}$所需的额外平均信息为：
$$
\text{KL}(p\lVert q)=-\int p(\textbf{x}) \ln q(\textbf{x}) \text{d}\textbf{x} - (-\int p(\textbf{x}) \ln p(\textbf{x}) \text{d}\textbf{x}) \\
= -\int p(\textbf{x}) \ln \frac{q(\textbf{x})}{p(\textbf{x})} \text{d}\textbf{x}.
$$
称为分布$p(\textbf{x})$和$q(\textbf{x})$间的*相对熵(relative entropy)或Kullback-Leibler divergence, KL divergence*. 这个值是不对称的，即$\text{KL}(p\lVert q)\not\equiv \text{KL}(q\lVert p)$.
KL-散度满足$\text{KL}(p\lVert q)\ge 0$，当且仅当$p(\textbf{x})=q(\textbf{x})$时取等号。根据凸函数的性质可以证明。

可以看到数据压缩和密度估计（即对一个未知概率分布进行建模的问题）之间的联系很紧密，因为只有得知真实分布的时候我们才能进行最有效的压缩。如果使用的分布与真实分布存在差距，则编码效率必然会下降，而平均增加的信息就等于两个分布间的KL-散度。
假设数据是在一个未知分布$p(\textbf{x})$中生成的，我们希望对这个分布进行建模。可以尝试使用一些通过参数$\theta$调节的参数分布$p(\textbf{x}|\theta)$来近似，例如多元高斯分布。一种选择$\theta$的方式就是最小化$p(\textbf{x})$和$p(\textbf{x}|\theta)$间关于$\theta$的KL-散度，但是并不能直接进行计算因为并不知道$p(\textbf{x})$.

对于两个变量的联合分布，如果两个变量相互独立，则联合分布等于两个边缘分布的乘积$p(\textbf{x,y})=p(\textbf{x})p(\textbf{y})$. 如果它们不独立，则可以通过计算联合分布和边缘分布乘积的KL-散度得到二者有多么“靠近”相互独立：
$$
\textbf{I[x,y]} \equiv \text{KL}(p(\textbf{x,y})\lVert p(\textbf{x})p(\textbf{y})) \\
= -\int \int p(\textbf{x,y})\ln \Big( \frac{ p(\textbf{x})p(\textbf{y})}{p(\textbf{x,y})} \Big) \text{d}\textbf{x}\text{d}\textbf{y}
$$
称为变量x与y间的*互信息(mutual information)*. 由KL-散度的性质可知，$I(\textbf{x}, \textbf{y})\ge 0$，当且仅当二者相互独立时取等号。互信息与相对熵之间的联系：
$$
\textbf{I[x,y]}=\textbf{H[x]}-\textbf{H[x|y]}=\textbf{H[y]}-\textbf{H[y|x]}
$$

可以将互信息视作当得知y的值时，x的不确定性的减少量（反之亦然）。从贝叶斯的角度，可以将$p(\textbf{x})$视作x的先验分布，将$p(\textbf{x|y})$视作观察新数据y之后的后验分布。这样的话互信息就代表了当观察到y之后x不确定性的减少量。


# Chapter 2. PROBABILITY DISTRIBUTIONS
*密度估计(density estimation)*：给定有限的观测集合，对随机变量的概率分布进行建模。
本章假设数据点都是独立同分布的。
需要强调，密度估计问题本质上是ill-posed的问题，因为对于同一个有限的观测集，有无数多个分布都可以得到。

离散随机变量的二项与多项分布以及连续型随机变量的高斯分布都是*参数(parametric)*分布的例子，叫这个名字是因为它们受一个较小的适应性参数支配，例如对于高斯分布的均值和方差。为了能够在密度估计中使用这些分布，需要在给定的观测数据集上确定这些参数的值。频率学派的处理方式是通过优化某种标准（例如似然函数）对这些参数选择特定的值。相比之下，贝叶斯的方法则引入参数的先验分布，然后使用贝叶斯理论来计算在给定数据下对应的后验分布。

*共轭(conjugate)*先验很重要，它可以使后验分布与先验分布有相同的泛函形式，因此就得到了简化的贝叶斯分析。例如多项分布参数的共轭先验称为*Dirichlet*分布，而高斯分布均值的共轭先验则是另一个高斯分布。所有这些分布都是*指数族(exponential family)*分布的例子，它们有一些重要的性质。
（如果先验分布和似然函数可以使得先验分布和后验分布有相同的形式，那么就称先验分布与似然函数是共轭的，共轭的结局是让先验与后验具有相同的形式。注意：共轭是指的先验分布和似然函数）

参数化方法的一个局限性是它对分布假设了一个特定的泛函形式，这也许不适用于某些特定的应用。另一个可选方法是*非参数(nonparametric)*密度估计方法，通常分布的形式依赖于数据集的大小。这种模型仍然包含参数，但是它们控制的是模型的复杂度而非分布形式。后边会讨论三种非参数方法，分别基于直方图、最近邻和核方法。

## 2.1. Binary Variables
Bernoulli分布：
$$
\text{Bern}(x|\mu)=\mu^x(1-\mu)^{1-x}
$$
其均值和方差为：
$$
\mathbb{E}[x]=\mu \\
\text{var}[x]=\mu(1-\mu)
$$

二项(binomial)分布：
$$
\text{Bin}(m|N,\mu)=\tbinom{N}{m}\mu^m(1-\mu)^{N-m}
$$
其中$\binom{N}{m}=\frac{N!}{(N-m)!m!}$
其均值和方差为：
$$
\mathbb{E}[m]=\sum_{m=0}^N m\text{Bin}(m|N,\mu)=N\mu \\
\text{var}[m]=\sum_{m=0}^N (m-\mathbb{E}[m])^2\text{Bin}(m|N,\mu)=N\mu(1-\mu)
$$

### 2.1.1 The beta distribution

Gamma分布：$\Gamma (x)=\int_0^{\infin}u^{x-1}e^{-u}\text{d}u$
性质：$\Gamma(x+1)=x\Gamma(x),\Gamma(1)=1$且当x是整数的时候$\Gamma(x+1)=x!$

如果选择一个与$\mu$和$(1-\mu)$成比例的先验，那么与先验和似然函数乘积成比例的后验分布，就会有与先验相同的形式。这个性质称为*共轭(conjugacy)*
依此选择一个先验称为beta分布：
$$
\text{Beta}(\mu|a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}
$$
其均值和方差为：
$$
\mathbb{E}[\mu]=\frac{a}{a+b} \\
\text{var}[\mu]=\frac{ab}{(a+b)^2(a+b+1)}
$$
参数a和b通常称为*超参数(hyperparameters)*因为它们控制参数$\mu$的分布。

## 2.2. Multinomial Variables
有K个互斥状态的离散变量

### 2.2.1 The Dirichlet distribution

## 2.3. The Gaussian Distribution
形式：
$$
\mathcal{N}(x|\mu, \sigma^2)=\frac{1}{(2\pi \sigma^2)^{1/2}}\exp{-\frac{1}{2\sigma^2}(x-\mu)^2}
$$
其中$\mu, \sigma^2$分别是均值和方差。
D维的多元高斯分布形式：
$$
\mathcal{N}(\textbf{x}|\mu ,\Sigma)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}\exp \Big\{ -\frac{1}{2}(\textbf{x}-\mu)^T\Sigma^{-1}(\textbf{x}-\mu) \Big\}
$$
其中$\mu, \Sigma, |\Sigma|$分别是D维的均值向量、$D\times D$的协方差矩阵和$\Sigma$的行列式

尽管高斯分布广泛地作为密度模型来使用，但它其实有一些缺陷。
首先，一个一般的对称的协方差矩阵$\Sigma$有$D(D+1)/2$个独立参数，加上$\mu$中的$D$个，总共有$D(D+3)/2$个。对于较大的D，其参数量增长是平方速度的。一种解决办法是对协方差矩阵加以限制，使其称为对角阵，或者干脆表示成单位阵的倍数形式。
另一个限制是，高斯分布本身是一个单峰的分布，因此无法较好地近似多峰分布。（当然都可以解决啦）

### 2.3.1 Conditional Gaussian distributions
多元高斯分布的一个重要性质是如果两组变量是联合高斯分布，那么其中一组变量关于另一组的条件分布同样也是高斯分布。（神奇嗷）

### 2.3.2 Marginal Gaussian distributions
已经知道如果一个联合分布是高斯分布，那么其条件分布也是高斯分布，实际上它的边缘分布同样也是高斯分布。


## 2.4. The Exponential Family
指数族的成员有很多通用的重要性质。
关于x分布的指数族在给定参数$\eta$的情况下，定义为具有下列形式的分布集合
$$
p(\textbf{x}|\eta)=h(\textbf{x})g(\eta)\exp\{\eta^T \textbf{u(x)}\}
$$
其中x可以是标量也可以是向量，可以是离散的也可以是连续的。$\eta$称为分布的自然参数。$\textbf{u(x)}$是$\textbf{x}$的某个函数。$g(\eta)$可以理解为使分布归一化的参数，满足：
$$
g(\eta)\int h(\textbf{x})\exp\{\eta^T \textbf{u(x)}\}d\textbf{x}=1
$$

### 2.4.2 Conjugate priors
一般来说，对于一个给定的概率分布$p(x|\eta)$，可以找到一个与似然函数共轭的先验函数$p(\eta)$，使后验分布和先验分布有相同的函数形式。

### 2.4.3 Noninformative priors
对于概率推断而言，有的时候有先验信息，而其他的时候则不知道分布的形式如何。于是有一种先验分布的形式称为无信息先验分布，它尽可能降低对后验分布的影响。有的时候也叫做“让数据自己说话”。
假设有一个分布$p(x|\lambda)$，由参数$\lambda$控制，也许能找到一个合适的先验分布$p(\lambda)=\text{const}$. 如果$\lambda$是一个有K个状态的离散变量，则先验概率分散到每个状态的概率为$1/K$. 而对于连续变量，则有两个难题；第一个是如果$\lambda$无界，则该先验分布无法正确归一化，因为关于$\lambda$的积分此时是发散的。这种先验称为improper.实际上，improper的先验通常可以用于提供对应的proper的后验分布，即可以正确归一化的（后验分布）。例如，如果将一个高斯分布的均值设定为一个均匀先验分布，那么当观测到至少一个数据点的时候，该均值的后验分布就是proper的。
第二个困难是当变量进行非线性变换的时候概率密度的变化。假设一个函数$h(\lambda)$是常数，如果将变量变成$\lambda=\eta^2$，那么$\hat{h}(\eta)=h(\eta^2)$同样也是常数。但是如果令密度函数$p_{\lambda}(\lambda)$是常数，则$\eta$的密度函数为：
$$
p_{\eta}(\eta)=p_{\lambda}(\lambda) | \frac{d\lambda}{d\eta} | =p_{\lambda}(\eta^2)2\eta\propto \eta
$$
这样的话$\eta$的密度就不会是常数。使用最大似然函数的时候不会遇到这样的困难，因为似然函数$p(x|\lambda)$是$\lambda$的简单函数，因此可以随意使用任何方便的参数。但是，如果要选择一个常数先验分布，就必须小心使用适当的参数表示。

这里考虑无信息先验的两个简单例子。第一个，如果一个密度函数的形式为：
$$
p(x|\mu)=f(x-u)
$$
那么参数$\mu$称为*位置参数(location parameter)*. 这一类密度函数展示了什么叫做*平移不变性(translation invariance)*，因为如果对$x$进行常数偏移得到$\hat{x}=x+c$，那么有：
$$
p(\hat{x}|\hat{\mu})=f(\hat{x}-\hat{\mu})
$$
其中定义$\hat{\mu}=\mu+c$.因此密度函数的形式不变，也就是说密度函数与原始变量的选取无关。我们希望选择一个能够反应这一平移不变性的先验分布，所以选择了一个先验，其
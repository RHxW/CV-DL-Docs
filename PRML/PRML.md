# Pattern Recognition and Machine Learning

## Chapter 1. Introduction
### 1.1
使用正则化方法减轻过拟合现象，在误差函数中引入一个正则化项以阻止权重参数值变很大，如：
$$
\tilde{E}(\textbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2+\frac{\lambda}{2}\lVert \textbf{w} \lVert^2  \qquad (1.4)
$$
其中w是权重向量($\lVert \textbf{w} \lVert^2 \equiv \textbf{w}^T \textbf{w} = w_0^2+w_1^2+...+w_M^2$)，$\lambda$是用来调整正则化项与其他部分比重的参数。
这个误差函数可以在闭式解下最小化。
这种技术在统计学上称为*收缩*方法(*shrinkage* methods)，因为它缩小了参数的值。
平方正则化方法称为*岭回归*(*ridge regression*)。在神经网络中这种方法称为*权重衰减*(*weight decay*)。

实际上$\lambda$控制了模型的有效复杂度，因此也决定了过拟合的程度。


### 1.2 Probability Theory
条件概率: $P(A|B)=\frac{P(AB)}{P(B)}$
有: $P(AB)=P(A|B)P(B)$
由对称性: $P(AB)=P(BA)$
得: $P(A|B)P(B)=P(B|A)P(A)$
整理得*Bayes' theorem*: $P(A|B)=\frac{P(B|A)P(A)}{P(B)}$
以及: 
$$
p(X)=\sum_{Y}p(X|Y)p(Y)
$$
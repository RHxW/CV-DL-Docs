# Intelligent Frame Selection as a Privacy-Friendlier Alternative to Face Recognition

广泛应用于人脸识别的监控摄像头引发了许多关于隐私的关注。本文提出一个用于大规模人脸识别的隐私友好型(privacy-friendly)的替换方案。与在全部视频数据上运行人脸识别软件不同，我们提出对每一个检测到的人自动提取一个高质量快照，而无需透露ta的身份。这张快照随后会被加密存储。

引入一个新的无监督人脸图片质量评估方法，用于选择高质量快照。使用公开数据集中的高质量人脸图片训练一个VAE(variational autoencoder，变分自编码器)，并将重构概率作为人脸质量估计的度量。



## 1 Introduction

隐私什么的blahblahblah

……

提取适合于人脸识别的人脸剪裁是一件困难的事，因为监控镜头一般都会受物体运动的影响而变模糊。而且，由于图片的质量受多个方面的影响（如角度，光照和头部姿态等），所以难以对单帧的质量进行量化。帧选择技术已经被用在人脸识别之前来减少计算量或者提高识别准确率。本作引入一个新的基于深度学习的技术来实现智能人脸识别。关键的贡献是一个新的人脸图片质量评估(FQA)方法，这个方法受异常检测(anomaly detection)启发。于之前的解决方案，我们的系统完全基于无监督训练且不需要人脸识别系统的参与。



## 2 Related work

blahblahblah



## 3 Frame selection as an alternative to face recognition

blahblahblah



## 4 Face image quality assessment

提出一种基于变分自编码器(VAE)的新技术用来进行人脸质量评估(FQA)。于其他FQA方法相比，此方法的优点在于完全无监督。不需要人脸识别系统或者数据集中id信息，且同样多训练集外数据泛化良好。

为了评估一个人脸图片的质量，计算一个使用高质量图片训练的VAE的重构概率。VAE的重构概率通常用作异常检测的度量，来判断输入与训练数据的差异程度。通过在公开数据集的高质量人脸图片上进行训练，我们将FQA任务重新表示为异常检测任务（也就是衡量当前人脸图片与训练中的高质量人脸图片的差异程度）

一个VAE是标准自编码器的一个基于概率的变体。编码器和解码器通过概率分布建模而非确定的函数。编码器$f_{\phi}(x)$对隐变量$z$的后验$q_{\phi}(z|x)$进行建模，解码器$f_{\theta}(z)$对数据$x$在给定隐变量$z$下的似然$p_{\theta}(x|z)$进行建模。选择高斯分布$\mathcal{N}(0,I)$作为隐变量$p_{\theta}(z)$的先验分布。后验$q_{\phi}(z|x)$和似然$p_{\theta}(x|z)$分别是各向同性多元正态分布$\mathcal{N}(\mu_z,\sigma_z)$和$\mathcal{N}(\mu_x,\sigma_x)$. Figure 3展示了一张图片$x$经过VAE的前向传播过程，每个箭头代表一个采样过程。要使用反向传播来训练VAE，则需要每个操作都是可微的，但是对于采样操作则不需要：$z\sim \mathcal{N}(\mu_z,\sigma_z)$和$\hat{x}\sim \mathcal{N}(\mu_x,\sigma_x)$. 使用re-parameterization技巧来解决这个问题。对一个专用的随机变量$\epsilon \sim \mathcal{N}(0,1)$进行采样，从而可以将采样操作重写成：$z\sim \mu_z + \epsilon \cdot \sigma_z$和$\hat{x}\sim \mu_x+\epsilon \cdot \sigma_x$. VAE训练的目标函数写作对数似然的期望减去后验和先验间的KL散度：
$$
\mathcal{L}(x)=E_{q_{\phi}(z|x)}(p_{\theta}(x|z))-KL(q_{\phi}(z|x)|p_{\theta}(z))  \qquad (1)
$$
第一项是重构项，强制生成输入$x$的一个好的重构$\hat{x}$. KL项通过强制其成为高斯分布来对隐空间的分布进行正则化。通过训练一个生成模型，如VAE，模型学习近似训练数据分布。当观测到一个较大的重构误差的时候，就意味着这个数据不是从VAE的训练数据的分布上进行采样得到的。

重构概率是重构误差通过将隐空间的变化和重构考虑在内得到的一种泛化形式。首先，将一张图片$x$送入编码器，生成均值向量$\mu_z$和标准差向量$\sigma_z$. 然后从隐分布$\mathcal{N}(\mu_z, \sigma_z)$采样$L$个样本$\{z^0,z^1,...,z^l\}$. 所有的样本$z^l$传入解码器得到$x$的重构分布（通过均值$\mu_{\hat{x}}^l$和标准差$\sigma_{\hat{x}}^l$描述）。重构概率是$x$在L个样本上平均的概率：
$$
\text{RP}(x)=\frac{1}{L}\sum\limits_{l=1}^L \mathcal{N}(x|\sigma_{\hat{x}}^l,\mu_{\hat{x}}^l)  \qquad (2)
$$
重构概率最初作为异常评分使用。当VAE仅使用“正常”数据进行训练的时候，隐分布学习将这些样本在一个低维空间进行表达。因此，“正常”数据中的样本会得到较高的重构概率，同时异常样本会得到较低的重构概率。将人脸图像的生物识别的质量作为使用高质量人脸图片训练的VAE的重构概率。对应的，高重构概率代表高人脸图片质量。注意到对于人脸图片质量没有明确的定义，而且质量得分独立于任何人脸识别模型。高质量人脸图像的定义是直接从训练数据中得到的。

编码器$f_{\phi}(x)$由5个连续的block构成，每个block内有一个卷积层，bn层和一个leaky ReLU激活函数以及最后两个全连接层。编码器的输出是定义$q_{\phi}(z|x)$的参数。解码器$f_{\theta}(z)$由5个block构成，每个block内有一个转置卷积层，bn层和一个leaky ReLU激活函数以及最后两个全连接层。解码器的输出是定义$p_{\theta}(x|z)$的参数。将$L$设定为10来计算重构概率。用CelebA数据集来进行训练。使用Adam算法，学习率为0.005，batch size为144. VAE训练了10个epochs，每张人脸都resize到64*64并转成了灰度图像。



## 5 Experimental setup

FQA的衡量指标——ERC(error vs. reject curve)，ERC衡量了在什么样的（质量）范围内，低质量样本能够增加认证(verification)的性能，认证性能通过FNMR(false non-match rate)进行衡量。FNMR是一个生物学特征匹配器将同一个体的两个信号误匹配成不同个体的几率。人脸认证由计算两张人脸图片的相似度评分和将这个相似度评分和一个阈值进行比较组成。
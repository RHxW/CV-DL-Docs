# Synthesizing Normalized Faces from Facial Identity Features

## Abstract

一种通过给定人脸输入图片生成一张正面、中性表情的人脸图片的方法。通过学习从人脸识别框架提取的特征生成人脸关键点和纹理来实现。与之前的生成方法不同，我们的编码特征向量对光照、姿态和面部表情有更强的不变性。利用这一不变性，只使用正面、中性表情的人脸图片训练解码器网络。由于这些图片都是对齐后的，我们可以将它们分解成一个关键点和对齐后纹理的稀疏集合。然后解码器分别预测关键点和纹理，并将二者通过一个可微的图片变形操作合并到一起。生成的图片可用于多种应用，例如分析面部属性、曝光和白平衡调整或者生成3D头像。



## 1. Introduction

在人脸识别领域，深度神经网络将一张人脸图片嵌入到高维特征空间，在其中同一个人的图片映射的点距离较近。诸如FaceNet等网络生成的特征向量对姿态、光照和表情的差异具有惊人的一致性。虽然对于神经网络而言，特征没什么特别的，但是对于人类来说特征难以理解。并没有什么显而易见的方法能够将特征反向生成人脸图片。

提出一个将人脸特征映射回人脸图片的方法。这个问题的约束很少：输出图像比FaceNet特征向量多了150多个维度。我们的关键思路是通过利用人脸特征对于姿态、光照和表情的不变性，将这个问题转换为特征向量映射到一个正向、中性表情脸，称为归一化人脸图像。直观上，身份identity到归一化人脸的映射差不多是一对一的，所以可以训练一个解码器网络来学习，如Figure 1所示。使用仔细构建的特征和归一化人脸对来训练解码网络。最好的结果是使用FaceNet的特征，但是公开的VGG-Face网络的特征效果也没差多少。

![Figure 1](1.png"Figure 1")

因为人脸特征是如此的可靠，训练好的解码器网络对很大范围的负面隐私，如遮挡、光照和姿态变化等都具有鲁棒性，甚至可以用于单色照片或者绘画上。网络的鲁棒性使其远胜于将人脸进行变形生成正脸的方法。

得到的归一化人脸的一致性使这种方法适用于一系列应用。例如，利用合成人脸的中性表情和关键点可以轻松地使用3D可变形模型创造一个虚拟现实头像。也可以通过变换输入图片的颜色来匹配预测的人脸来实现自动颜色校正和白平衡。还可以将我们的方法当作一种研究的工具来可视化哪些特征被人脸识别系统可靠地捕捉到了。

与active shape model类似，我们的解码器网络明确地将人脸的几何特性和纹理特性进行了解耦。在我们的实现中，解码器同时生成注册的纹理图片和面部关键点位置作为中间级的活动。基于关键点，将纹理进行形变处理来得到最终的图片。

解决了几个技术挑战。首先，端到端学习需要变形操作是可微的。使用一种高效又易于实现的基于样条插值的方法。这个方法允许计算输入图片和输出图片间的FaceNet特征相似度，将其作为训练目标，辅助获得感官相关的细节。

其次，难以获得大量的正脸、中性表情训练集。我们采用了一种数据增广范式作为解决方法，这种范式利用了纹理-形状分解。

blahblahblah



## 2. Background and Related Work

### 2.1. Inverting Deep Neural Network Features

对于理解深度网络预测结果的兴趣引发了多种根据特定特征向量创建图像的方法。其中一个直接通过梯度下降来优化图像像素。由于与特征相关的像素空间过于庞大，优化过程需要较强的归一项，例如全变分或者高斯模糊。生成的图片很有意思但并不真实。

另一种联系更紧密的方法通过训练一个前馈网络将给定embedding进行反转。Dosovitskiy and Brox将这个问题视为构建给定特征的最可能的图片。与它们相比，我们的方法采用更严格的惩罚使生成的图片必须是一张归一化人脸。



### 2.2. Activate Appearance Model for Faces

wraping operations

blahblahblah



### 2.3. FaceNet

128维特征向量 

blahblahblah



### 2.4. Face Frontalization

blahblahblah



### 2.5. Face Generation using Neural Networks

blahblahblah



## 3. Autoencoder Model

假设训练集是一组中性表情的正脸图片。前处理是将每张图片分解成纹理$T$和一组关键点$L$，使用的方法是现成的关键点检测工具和第四节中介绍的形变技术。

测试的时候，考虑到图片来源于非限制场景，不适合使用训练时的前处理流程。则使用深度架构直接从图像映射到$L$和$T$的估计。整个网络架构如Figure 3所示。

![Figure 3](3.png"Figure 3")

### 3.1. Encoder

编码器接收一个输入图像$I$并返回一个$f$维的特征向量$F$. 要保证编码器对图片的域差异具有鲁棒性。假设人脸识别模型能够去除掉人脸图片中与身份无关的信息。这样，在控制下的训练图像就能够和非限制场景图象映射到同一个空间。这就允许我们只在控制图像上训练就可以。

### 3.2. Decoder

可以使用深度网络直接从$F$映射得到输出图片。这样做需要对面部的几何和纹理变量同时建模。一种更高效的方法是分别生成关键点$L$和纹理$T$，然后使用形变操作渲染最终结果。

在$F$上使用一个带ReLU的浅MLP生成$L$. 使用CNN生成纹理图片。先用一个全连接层将$F$映射到$14\times 14\times 256$的局部化特征。然后使用一组转置卷积，卷积核为5，步长为2，和ReLU一起上采样到$224\times 224\times 32$的局部化特征。第$i$个转置卷积层后的通道数为$\max(256/2^i,32)$. 最后使用一个$1\times 1$卷积得到$224\times 224\times 3$的RGB值。

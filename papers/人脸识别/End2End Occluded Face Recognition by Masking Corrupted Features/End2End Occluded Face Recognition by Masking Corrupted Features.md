# End2End Occluded Face Recognition by Masking Corrupted Features

## 1 Introduction
提出一个针对鲁棒遮挡人脸识别的简单高效的方法用于清除网络中损坏的特征，使用的是特征mask.提出的方法叫FROM(Face Recognition with Occlusion Masks)，采用一个子网络（即Mask Decoder）来动态地解码精确的特征mask，其指出了由遮挡引起的损坏位于特征中的位置，并且通过将mask和特征相乘的方式对损坏部分进行清除。另外，提出利用遮挡相互接近的特点（例如相邻的人脸部分通常遮挡状态类似）作为masks学习的监督。Contributions are:
1. 提出一个单一网络的遮挡人脸识别方法——FROM，它可以通过端到端的方式进行训练，同时学习到特征的masks和对遮挡鲁棒的深度特征。
2. 提出利用一个Mask Decoder来学习一个“遮挡到特征mask”的映射，接收特征图金字塔作为输入，同时捕捉到局部和全局的信息来学习到精准的mask. 而且随后的Occlusion Pattern Predictor会引导Mask Decoder生成与遮挡模式一致的masks.

## 2 Related Work
...
![Figure 2](2.png 'Figure 2')
## 3 Method
提出的FROM的架构如Figure 2所示。接收一组随机遮挡以及无遮挡的人脸图形作为输入，生成特征金字塔，随后用于解码特征masks，然后用获取的masks通过与特征相乘的方式对特征进行清洗，清除掉损坏的部分，结果用于最终的识别任务。FROM方法的核心思想是学习精确的特征mask来有效地清除损坏特征。为了实现这一想法，将输入人脸图片的遮挡信息的监督加入到解码masks的学习中，在输入的遮挡模式和解码的masks间建立了联系。设计了一个关于遮挡模式的特殊形式的监督，将遮挡相近的特点考虑其中，使Mask Decoder的学习过程更稳定且能的高更准确的masks从而实现更高的准确率。

### 3.1 Deep General Face Recognition
人脸识别的训练一般采用基于边界的softmax损失函数，我们使用的损失函数形式为：
$$
\mathcal{L}_{margin}=-\frac{1}{N}\sum_{i=1}^N \log \frac{e^{s\cdot \delta}}{e^{s\cdot \delta}+\sum_{j\ne y_i}^n e^{s\cos \theta_j}} \\
\delta=\cos(m_1\theta_{y_i}+m_2)-m_3
$$
在$(m_1,m_2,m_3)$超参数的设定上，SphereFace，ArcFace和CosFace的设置分别为$(m_1,0,0),(0,m_2,0),(0,0,m_3)$
### 3.2 Feature Pyramid Extractor
为了能够同时获取学习精确mask所需的空间感知特征和识别所需的判别特征，采用Feature Pyramid Extractor作为backbone网络，实际上采用的是ArcFace的LResnet50E-IR作为backbone的主体。如Figure 2所示，对齐后人脸图片作为输入，输出金字塔特征X1,X2和X3.注意到X1是需要进行清洗的判别特征，尺寸为$N\times C \times H \times W$.将包含局部和全局信息的X3输入Mask Decoder解码得到对应的特征mask M用于去除X1中的损坏的元素。
![Figure 3](3.png 'Figure 3')
### 3.3 Mask Decoder
FROM的主旨是学习一个Mask Decoder用于生成一个特征mask来清除特征中由遮挡引起的损坏。如Figure 2所示，它从X3中解码出遮挡信息作为mask M.然后对X1进行清洗，用清洗后的特征进行识别。如Figure 3a所示，Mask Decoder的结构较为简单，其中的Sigmoid函数用于将输出的特征mask限制在0-1之间。
#### 3.3.1 Mask Source:middle vs. deep
使用Feature Pyramid Extractor同时利用局部和全局特征来获取mask，如Figure 2所示
#### 3.3.2 Mask Location:conv vs. fc
fc会丢失空间信息，所以在fc之前前进行mask（废话么）
#### 3.3.3 Mask Dimension:3D vs. 2D
3D（本质上就是空间注意力机制）
#### 3.3.4 Mask Format:dynamic vs. static
有的方法采用固定的mask用于去除损坏特征，本作采取动态mask的形式

### 3.4 Occlusion Pattern Predictor
引入Occlusion Pattern Predictor来监督mask的学习。如Figure 2所示，它接收mask作为输入，并预测遮挡模式向量，然后用softmax loss进行分类。其结构如Figure 3b所示。
Mask Decoder训练生成的mask应该有两个特点：
1. 与输入图片的遮挡相关
2. 能够正确地mask出对人脸识别有害的部分
第一个特点就是通过引入Occlusion Pattern Predictor来进行监督的，第二个经过人脸识别任务监督
#### 3.4.1 Proximate Occlusion Patterns
人脸图片分成$K\times K$个块，每个块代表了一个人脸的子区域，而每个人脸的子区域都有可能被遮挡。实际上如果一张人脸图像被分成$K\times K$个块，那么就有$2^{K\times K}$中遮挡模式。
这里提出一个更合理的近似方法来解决上述维度问题。观察到在实际应用中，相邻的block通常遮挡情况相近，参考这一特点，通过将遮挡模式限制为覆盖相邻$m\times n$个block的矩形从而降低可能的遮挡模式的数量。
![Figure 4](4.png 'Figure 4')
Figure 4中展示了K=4的时候的几种m和n的组合。例如红色的模式包含$1\times 2$个相邻的blocks，黄色的模式包含$3\times 3$个相邻的blocks. 图中的数值矩阵代表不同尺寸模式的数量。
对于干净的图片（无遮挡），将其对应的模式视为覆盖$0\times 0$的blocks.这种方式使FROM无需依赖于额外的网络来判断图片是否有遮挡。对于无遮挡的图片，Mask Decoder会输出不影响原特征的masks.
一般来说，对于划分成$K\times K$个blocks的图片，有$(K\times (K+1)/2)^2+1$种遮挡模式。
#### 3.4.2 Pattern Prediction

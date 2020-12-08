# NOTE


## Cross-pose Identifying faces across a wide range of poses: 

1. TP-GAN[91]: <Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis>, 2017 GAN 从侧脸生成正脸图像

2. PIM[324]: <Towards Pose Invariant Face Recognition in the Wild>, 2018 GAN，与TP-GAN类似
3. DREAM[22]: <Pose-Robust Face Recognition via Deep Residual Equivariant Mapping>, 2018 , 针对侧脸识别，提出DREAM模块
4. DA-GAN[344]: <Dual-Agent GANs for Photorealistic and Identity Preserving Profile Face Synthesis>, 2017 GAN, 用3D模型辅助合成人脸 
5. DR-GAN[227]: <Disentangled Representation Learning GAN for Pose-Invariant Face Recognition>, 2018 GAN, 接收一张或多张图片生成一个人脸表示
6. UV-GAN[43]: <UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition>, 2017 GAN, 不同姿态生成UV map
7. CAPG-GAN[88]: <Pose-Guided Photorealistic Face Rotation>, 2018 GAN, 人脸图像加关键点生成不同姿态的人脸图像。（目测对表情不鲁棒，而且要求关键点准）
8. PAMs[160]: <Pose-Aware Face Recognition in the Wild>, 2016, 非配合场景下人脸识别方法
9. AbdAlmageed et al.[1]: <Face recognition using deep multi-pose representations>
10. MvDN[107]: <Multi-view deep network for cross-view classification>



## Specialized architectures.

1. aaa

2. Bilinear CNN[134]: <Bilinear CNNs for Fine-grained Visual Recognition>,  2015, 双通道（两个cnn）提特征，然后两个特征相乘（？？？）

3. Chowdhury et al.[34]: <One-to-many face recognition with bilinear CNNs>, 2016, Bilinear CNN用与人脸识别

4. Comparator Networks[276]: <Comparator Networks>, 2018, 用于判断两组图片是否为同一个东西的end2end网络

5. Han et al.[75]: <Face Recognition with Contrastive Convolution>, 2018, 

   > 基于卷积神经网络设计的人脸识别模型在进行人脸验证的时候，首先独立地提取待比对的两张人脸的特征。由于所有的人脸在提取特征时都是利用相同的，不变的网络参数，对于给定的一张人脸，它和任意一张人脸对比时，它的特征都保持不变。而我们人类在比对一对人脸时，对一张人脸的特征关注会随着另一张人脸的特征的变化而变化。这一现象启发我们设计一种新型的网络结构用于人脸识别，也就是本文提出的带对比卷积的人脸识别模型。通过精心的设计，对比卷积的卷积核只关注当前待比较的两张人脸的对比特征，也就是这两张人脸的差异特征。通过可视化对比卷积后的特征图，我们证实了该网络结构和动机的一致性，同时，我们在LFW和IJB-A上验证了该方法的有效性。”

6. Kang et al.[109]: <Pairwise relational networks for face recognition>, 2018,  目前人脸特征的缺陷：不知道使用了特征的哪部分，特征的哪部分是有意义的，特征的哪部分是可分离的、具有分辨能力的。因此也就难以得知什么样的特征可以将不同id的人脸图片清晰地分开。为了克服这一限制提出一个新的人脸识别方法，称为成对关系网络(PRN, pairwise relational network)，来捕捉同一id图片独特的关系并区分不同id间图片的关系。

7. AFRN[108]: <Attentional Feature-Pair Relation Networks for Accurate Face Recognition>, 2019, 使用注意力机制提升pairwise relation network效果

8. FANFace[283]: <FAN-Face: a simple orthogonal improvement to deep face recognition>, 2020, 

   > 众所周知，面部关键点可以提供姿态、表情和形状信息。而且当进行人脸匹配的时候，例如一张侧脸或有表情面部图片和一张正脸图片匹配的时候，关于这些关键点的知识可以用来确定对应关系，从而提升识别效果。但是，在之前的人脸识别方法中，只将关键点用于裁切以去除尺度、旋转和平移的影响。本文提出一个简单的人脸识别方法，其逐步将一个人脸关键点定位网络不同层的特征整合到识别网络的不同层中。为了实现这一目标，提出一个合适的特征整合层，在整合前使特征兼容。

9. PFE[199]: <Probabilistic Face Embeddings>, 2019, 
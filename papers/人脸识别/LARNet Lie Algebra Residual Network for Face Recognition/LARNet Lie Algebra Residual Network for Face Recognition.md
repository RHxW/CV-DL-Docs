# LARNet Lie Algebra Residual Network for Face Recognition
## Abstract
本文提出一个基于李代数的新方法，探索了人脸在3D空间的旋转如何影响CNN生成的特征。证明了人脸在图片空间的旋转等同于CNN在特征空间的一个额外的残差部分，该部分仅受旋转的影响。基于理论分析提出了LARNet(Lie Algebraic Residual Network)用于解决跨姿态人脸识别问题，LARNet的构成：
1. 一个用于从输入图片中解码旋转信息的残差子网络
2. 一个通过学习旋转量级来控制残差部分对特征学习贡献强度的门控子网络

## 1. Introduction

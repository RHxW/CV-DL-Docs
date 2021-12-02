# End-to-End Object Detection with Transformers
## Abstract
提出一个新方法，将目标检测看作一个直接集合预测问题。本方法去掉了传统目标检测流程中的很多人工设计的部分，例如nms或者anchor生成过程。这个新框架的主要部分称为DETR(DEtection TRansformer)，它是一个set-based的全局loss，它强制预测结果经过两部分匹配，以及一个transformer 编码解码结构。给定一个关于学习到的物体序列的固定的小集合，DETR推理出物体间以及全局图片上线问间的关系，直接并行输出最终的预测集合。

## 1 Introduction

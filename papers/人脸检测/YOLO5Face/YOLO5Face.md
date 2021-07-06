# YOLO5Face

Figure 1(d)引入Stem结构是一个创新点，用Stem结构替换原YOLOv5中的Focus层。

关键修改：
* 在YOLOv5网络中加入关键点回归head，并使用Wing loss作为loss函数。这个额外的监督可以提升检测器的精度
* 将YOLOv5中的Focus层替换未Stem block结构。提升了网络的泛化能力，并在性能不下降的前提下减少了计算复杂度。
* 修改SPP block，使用更小的池化kernel. 使YOLOv5更适合人脸检测并提升检测精度
* 加入了步长为64的P6输出block. 这样提升了检测较大尺寸人脸的能力。
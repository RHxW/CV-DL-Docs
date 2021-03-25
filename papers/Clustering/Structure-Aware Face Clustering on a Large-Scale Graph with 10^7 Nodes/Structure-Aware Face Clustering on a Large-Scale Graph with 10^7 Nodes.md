# Structure-Aware Face Clustering on a Large-Scale Graph with 10^7 Nodes

## Abstract

STructure-AwaRe Face Clustering（STAR-FC）方法，设计了一个能够保留结构信息的子图采样策略，用于探索大规模训练数据的能量，可将训练数据的尺度从1e5提升至1e7. 推理的时候STAR-FC在整个图上进行聚类，两个步骤：图解析(graph parsing)和图微调(graph refinement). 第二步引入节点亲密度(node intimacy)来挖掘局部结构信息。
# A Morphable Model For The Synthesis Of 3D Faces

一共有m个人脸，每个人脸的三维信息中有n个顶点
形状向量$S=(X_1,Y_1,Z_1,X_2,Y_2,Z_2,...,X_n,Y_n,Z_n)^T\in \mathbb{R}^{3n}$，n个顶点的三维坐标
纹理向量$T=(R_1,G_1,B_1,R_2,G_2,B_2,...,R_n,G_n,B_n)^T\in \mathbb{R}^{3n}$，n个顶点的RGB值

使用m个样本构建人脸模型，可以表示为m个样本的线性组合：
$$
\textbf{S}_{mod}=\sum_{i=1}^m a_i \textbf{S}_i,
\textbf{T}_{mod}=\sum_{i=1}^m b_i \textbf{T}_i,
\sum_{i=1}^m a_i=\sum_{i=1}^m b_i=1
$$

对人脸形状和纹理进行PCA：
1. 计算$\bar{S}, \bar{T}$
2. 中心化：$\Delta S=S - \bar{S},\Delta T=T - \bar{T}$
3. 计算协方差矩阵$C_S,C_T$
4. 求特征值和特征向量$(\alpha, s),(\beta, t)$

则：
$$
S_{model}=\bar{S}+\sum_{i=1}^{m-1}\alpha_i s_i,
T_{model}=\bar{T}+\sum_{i=1}^{m-1}\beta_i t_i,
$$
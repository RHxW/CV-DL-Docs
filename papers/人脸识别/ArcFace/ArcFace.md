# ArcFace: Additive Angular Margin Loss for Deep Face Recognition

广泛使用的分类loss——softmax loss的形式如下：
$$
L_1=-\frac{1}{N}\sum^N_{i=1}\log \frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^n e^{W_j^Tx_i+b_j}}
$$

简单起见，将$b_j$固定为0，然后对logit进行变换$W_j^Tx_i=\lVert W_j\rVert \lVert x_i\rVert \cos \theta_j$,其中$\theta_j$是权重$W_j$和特征向量$x_i$间的夹角。通过l2归一化对权重进行归一化$\lVert W_j\rVert=1$.同样对$x_i$使用l2归一化并re-scale到s.归一化之后的预测就仅依赖于权重和特征之间的夹角了。因此，学习到的特征向量就分布在一个以s为半径的超球面上。
$$
L_2=-\frac{1}{N}\sum^N_{i=1}\log \frac{e^{s\cos\theta_{y_i}}}{e^{s\cos\theta_{y_i}}+\sum_{j=1,j\neq y_i}^n e^{s\cos\theta_j}}
$$

由于特征向量在超球面上围绕各自的类中心向量分布，所以在权重和特征间加入一个角度边界惩罚项m来同时增强类内紧凑程度和类间分离度。
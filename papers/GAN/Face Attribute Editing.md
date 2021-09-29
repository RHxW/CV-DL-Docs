# Face Attribute Editing/Manipulating

## What is Attribute Actually?
人脸属性，比如眼睛大小、肤色、性别、人种等，通常我们会认为这些属性是独立且不可分的（也许用‘原子性’这个说法更好理解）。当然这是最理想的情况，比如我们希望在调节脸型的同时不改变嘴巴大小或者眼睛的位置。但实际上人脸的各个属性之间总存在着各种各样的联系（比如人脸姿态和五官尺寸之间的关系），而且其中的绝大多数是我们人类无法直接观测到的，所以希望人脸属性间的*完全*分离是不现实的，我们只能不断推进它们在一定范围内尽可能地分离。
而对人脸属性进行划分这个操作，其本质是在人脸特征的空间（隐空间）中寻找一组基或者称为主轴(basis, main axis)；那么对特征尽可能分离的期望就对应为希望这组基尽可能地正交。
## StyleGAN
StyleGAN引入了一个mapping network从而获得一个latent space，引入mapping network的目的是希望在中间latent space中生成的属性能够更好的分离/解耦，但是StyleGAN本身并没有提供分离的特征或对应方法。
那么如何证明引入的mapping network是有效的呢？
在StyleGAN论文的第四章节'Disentanglement studies'中就讨论了如何对特征的分离性进行衡量，从而证明了mapping network的有效性。
其中提到了两种度量方法：
1. **Perceptual path length** 基于‘属性越分离，线性插值效果越好’的假设。对某个属性进行线性插值，然后评估生成图片变化的剧烈程度，作为监督。评估的方法为perceptual distance
2. **Linear separability** 基于‘解耦程度越高越正交’的假设。对二元属性进行分类做监督。
这两种度量方式分别对应了特征空间的线性程度和基的正交程度
作者对mapping network的实验表明：加入mapping network可以有效提升中间latent space的属性分离度，但是会降低原始空间的属性分离度。
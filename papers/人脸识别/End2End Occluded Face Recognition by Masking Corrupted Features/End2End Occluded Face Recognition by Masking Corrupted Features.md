# End2End Occluded Face Recognition by Masking Corrupted Features

## 1 Introduction
提出一个针对鲁棒遮挡人脸识别的简单高效的方法用于清除网络中损坏的特征，使用的是特征mask.提出的方法叫FROM(Face Recognition with Occlusion Masks)，采用一个子网络（即Mask Decoder）来动态地解码精确的特征mask，其指出了由遮挡引起的损坏位于特征中的位置，并且通过将mask和特征相乘的方式对损坏进行清除。
# 照片风格迁移思路

>原始图片+风格化图片（油画／素描啥的）生成 具有风格化图片特征的原始图像内容。



![](https://upload-images.jianshu.io/upload_images/1271438-2f6bcd5124ca6267.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


两张图像经过预训练好的分类网络，若提取出的高维特征之间的欧氏距离越小，则这两张图像内容越相似

两张图像经过与训练好的分类网络，若提取出的低维特征在数值上基本相等，则这两张图像风格越相似；
换句话说，两张图像相似等价于二者特征的Gram矩阵具有较小的弗罗贝尼乌斯范数。

>格拉姆矩阵可以看做feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵），在feature map中，每个数字都来自于一个特定滤波器在特定位置的卷积，因此每个数字代表一个特征的强度，而Gram计算的实际上是两两特征之间的相关性，哪两个特征是同时出现的，哪两个是此消彼长的等等，同时，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram有助于把握整个图像的大体风格。有了表示风格的Gram Matrix，要度量两个图像风格的差异，只需比较他们Gram Matrix的差异即可。
总之，格拉姆矩阵用于度量各个维度自己的特性以及各个维度之间的关系。内积之后得到的多尺度矩阵中，对角线元素提供了不同特征图各自的信息，其余元素提供了不同特征图之间的相关信息。这样一个矩阵，既能体现出有哪些特征，又能体现出不同特征间的紧密程度。

具体原理参见:

* [浅谈协方差矩阵](https://www.cnblogs.com/chaosimple/p/3182157.html)
* [「协方差」与「相关系数」](https://www.zhihu.com/question/20852004/answer/134902061)
* [Gram矩阵](https://blog.csdn.net/wangyang20170901/article/details/79037867)


参考项目：
* https://blog.csdn.net/qq_25737169/article/details/79192211


## 使用VGG网络提取图片特征

>VGG是一个良好的特征提取器，其与训练好的模型也经常被用来做其他事情，比如计算perceptual loss(风格迁移和超分辨率任务中)，尽管现在resnet和inception网络等等具有很高的精度和更加简便的网络结构，但是在特征提取上，VGG一直是一个很好的网络

参考代码【inception-v3迁移学习】https://github.com/xander-ye/deep_learn/tree/master/tensorflow_1.4.0

```
inception_v3_transfer_learn_datasets.py
inception_v3_transfer_learn_inferance.py
inception_v3_transfer_learn_train.py
```
主要是**inception_v3_transfer_learn_datasets.py**中，使用inception-v3训练好的模型提取指定层特征

[tensorflow预训练模型下载(含vgg)](https://github.com/tensorflow/models/tree/master/research/slim)












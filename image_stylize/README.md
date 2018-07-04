# 照片风格化思路

>原始图片+风格化图片（油画／素描啥的）生成 具有风格化图片特征的原始图像内容。定义style_loss、content_loss，将这两种loss以一定的比例加起来作为最终的loss


![](https://upload-images.jianshu.io/upload_images/1271438-92ceb980c286f761.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 神经网络不用再次训练，调整的是白板图片内容，使得loss最小

参考项目：https://github.com/anishathalye/neural-style
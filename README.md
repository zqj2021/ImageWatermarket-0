# 数字图像水印

#### 介绍
仓库复现了文章：An Automated and Robust Image Watermarking Scheme Based on Deep Neural Networks 提出的模型
实现了基于深度卷积网络的鲁棒图像水印


#### 运行环境
实验环境 Tesla M40 24G
Windows系统
Tensorflow 2.8.0 版本
Cuda 11.1

### 模型框架
原论文模型描述
![原论文模型](https://foruda.gitee.com/images/1688297755015777858/6d5f9f9c_12398249.png "屏幕截图")
文中残差块实现
![残差块实现](https://foruda.gitee.com/images/1688297790303249179/19ba4242_12398249.png "屏幕截图")
代码实现模型，其中model_x为残差块模型（由Tensorboard生成）
![代码实现模型，由Tensorboard生成](https://foruda.gitee.com/images/1688298039961008682/b54601ea_12398249.png "屏幕截图")
残差块实现
![残差块模型实现](https://foruda.gitee.com/images/1688298143029771866/f7178634_12398249.png "屏幕截图")

### 使用说明
0. 实验数据集为 COCO、ImageNet、 Cifar10、QRCode
1. 安装环境
2. 运行model.py 文件执行代码
3. 修改参数在train_config.py文件
4. Tensorboard 监视运行环境，log目录为save_data/logs文件夹


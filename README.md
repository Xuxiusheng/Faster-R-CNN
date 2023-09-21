## **Faster-RCNN-Pytorch**

- 链接：https://pan.baidu.com/s/1GWatWKEJx_BVRK1liJ9rsw 
  提取码：xvg3 
- 这是我训练的模型，在VOC2007上的mAP是48%左右，只训练了20个轮次

## **环境配置**

- **Python 3.8**
- **Pytorch 1.11.0**
- **Windows**

## 文件结构

```
├── models: 网络结构定义，包括主干网络、RPN和检测头
├── utils: 训练验证模块以及预处理模块
├── dataset.py: 数据加载器模块
├── trainer.py: 模型训练器
├── train.py: 训练入口
```

## **数据集**

- 数据集分为训练集和测试集，数据信息保存在txt中
- txt中每一行代表一张图片的信息，包括路径、坐标和类别信息
- 每一行的具体信息包括：图片路径 xmin,ymin,xmax,ymax,class,如果有多个目标框以空格分隔

## **训练方法**

- 参数配置

  - 必改参数：
    - classes_dir: 设置成自己的种类文件路径
    - test_gap: 计算mAP的周期，如果数据集比较大，设置的可以大一些，默认是10个epoch
    - save_dir: 模型的保存路径

  - 非必改参数
    - input_shape: 模型训练输入尺寸，电脑性能好的话，可以设置的大一些
    - bs：训练批次，同上，根据电脑性能设置
    - id: 训练使用的GPU设备，目前只支持单GPU训练
    - pretrain_backbone: 是否使用冻结策略，即先冻结主干网络，只训练RPN和Head，默认使用
    - freeze_epoch: 冻结的轮次，pretrain_backbone为True时，此参数才有效

- Anchor尺寸

  先验Anchor的尺寸可以通过KMeans确定，也可以手工设定，本项目直接手工对Anchor硬编码，如果使用的数据集对象尺寸和VOC数据集相差比较大的话，建议使用KMeans聚类

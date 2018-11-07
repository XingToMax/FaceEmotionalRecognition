# FaceEmotionalRecognition
基于深度学习的表情情绪模型

## 基本说明

本系统是通过`tensorflow`实现的表情识别模型，支持识别如下几种表情(数据集为fer2013)

```
angry
disgust
fear
happy
sad
surprise
neutral
```

## 目录结构

+ dataset_helper

  包含一些数据集处理的脚本，包括将fer2013转换到tfrecords

+ images

  readme使用图片

+ inference

  网络结构

+ models

  训练好的模型

+ res

  opencv的人脸检测器

+ test

  模型测试

+ train

  模型训练

+ app.py

  系统入口

+ labels.py

  标签定义

## 系统主要依赖

+ python3.5
+ tensorflow 或 tensorflow-gpu
+ opencv-python

## 一些系统启动命令

执行app.py(需下载完成已经训练好的模型)

``` python
# 开启实时视频识别模式
python run.py -v
# 识别单张图片
python run.py -p path(图片路径)
# 训练
python run.py -train
# 测试
python run.py -test
```

训练直接运行train\main_train.py(需下载完成训练集)

测试直接运行test\main_test.py(需下载完成测试集)

## 关于模型及数据集的获取

网盘地址 : https://pan.baidu.com/s/1FVna89oPvi4PiY-voMwEEA 提取码: knf4

下载完成后直接在本项目根目录解压
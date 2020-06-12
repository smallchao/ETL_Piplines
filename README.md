# ETL_Piplines
Udacity 灾害应对管道项目

## 一、项目背景

有效的信息传递中一个关键点是信息被传送给正确的接受方，这在灾难相应过程中尤为重要。当发生时，工作人员需要对消息进行分类以送达正确的灾难应急机构。本项目就起到了这么一个作用，可以对消息进行分类。

## 二、项目结构

#### 1.ETL 管道

process_data.py是一个数据清洗管道，实现了如下功能：
- 加载 messages 和 categories 数据集
- 将两个数据集进行合并 (merge)
- 清洗数据
- 将其存储到 SQLite 数据库中

#### 2.机器学习管道

train_classifier.py是一个机器学习管道，实现了如下功能：

- 从 SQLite 数据库中加载数据
- 将数据集分成训练和测试集
- 搭建文本处理和机器学习管道
- 使用 GridSearchCV 对模型进行训练和微调
- 输出测试集的结果
- 将最终的模型输出为 pickle 文件

#### 3.Flask 网络应用程序

run.py是一个基于Flask的网络应用程序，用于最终的成果展示

## 三、其他文件

#### 1.数据源

messages.csv 和 categories.csv 数据集

#### 2.pipline构建过程

ETL Pipeline Preparation-zh.html
ML Pipeline Preparation-zh.html

#### 3.模型文件

model.sav
# NetEase_task---
kaggle  Porto Seguro’s Safe Driver Prediction

本代码为网易大作业，原型模型训练的完成内容，共有三个文件夹

1.data  代表源数据集，由于数据集较大这里并没有上传，如需下载，请在如下链接下进行下载

https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

2.code  文件夹下为所有的代码，其中，

kaggleEvalFunc.py 为本次题目所用到的损失函数的代码

LightGbmModel.py  为LightGbm模型的代码

DnnModel.py       为Dnn模型的代码

nn_embedding.py   为DNN_embedding模型的代码

libffmTXTGen.py   为libffm训练集和测试集文件的产生代码

ffm_run.py        为libffm模型的运行代码

model_ensemble.py 为stacking部分的代码

main.py           会对以上几个部分的模型的代码依次进行运行并产生结果

3.output  为保存输出结果的文件夹

由于输出的结果文件较大，因此没有上传。

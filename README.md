
# VideoCaption
视频的文本摘要(标注)，输入一段视频，通过深度学习网络和人工智能程序识别视频主要表达的意思(Input a video output a txt decribing the video)。

本程序总共包含3个模块：
(1). 视频读取与关键帧提取模块
(2). Image caption模块(通过训练一个CNN feature extracter + LSTM网络)
(3). Text summary模块

依赖包:
python3, numpy, opencv, pytorch, jieba分词, textrank4zh, tdqm

预训练的模型文件：
Image caption模块的模型文件和Text summary模型文件，链接：

使用方法：
python videoCaption.py video_file

脚本运行效果截图：
![](https://github.com/CaptainEven/VideoCaption/blob/master/screen%20shots/result.png)

算法主要步骤和脚本文件详解：
(1). ShortDetector文件:
  通过opencv读入视频流，使用3帧间差法计算相邻2帧的直方图帧间一阶差分和二阶差分算子,然后根据阈值判断是否切换镜头，保存镜头所在的帧ID即可。
  视频文件较大也可以，程序会将视频文件通过流的方式逐步读入内存。
(2). Imge Caption模块:
 


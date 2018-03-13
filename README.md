  
# VideoCaption
视频的文本摘要(标注)，输入一段视频，通过深度学习网络和人工智能程序识别视频主要表达的意思(Input a video output a txt decribing the video)。
</br>
</br>
本程序总共包含3个模块：</br>
(1). 视频读取与关键帧提取模块 </br>
(2). Image caption模块(通过训练一个CNN feature extracter + LSTM网络) </br>
(3). Text summary模块 </br>

依赖包:
python3, numpy, opencv, pytorch, jieba分词, textrank4zh, tdqm
</br>
预训练的模型文件：
Image caption模块的模型文件和Text summary模型文件，链接：
</br>
使用方法：
python videoCaption.py video_file
</br>
脚本运行效果截图：(测试的视频是薛之谦的《演员》mv)
![](https://github.com/CaptainEven/VideoCaption/blob/master/screen%20shots/result.png)
</br>
算法主要步骤和脚本文件详解：</br>
(1). ShortDetector文件:
  通过opencv读入视频流，使用3帧间差法计算相邻2帧的直方图帧间一阶差分和二阶差分算子,然后根据阈值判断是否切换镜头，保存镜头所在的帧ID即可。
  视频文件较大也可以，程序会将视频文件通过流的方式逐步读入内存。</br>
(2). Imge Caption模块:
  本模块将深度卷积神经网络和深度循环神经网络结合，用于解决图像标注和语句检索问题。通过CNN提取输入图像的高层语义信息，然后输入到LSTM不断预测下一个最可能出现的词语，组成图像描述。训练的目标就是输出的词语与预期的词语相符合，依次设计神经网络的loss函数。本程序提供训练好的模型，链接见上。读者想要用自己的数据及训练也是可以的。</br>
(3). 文本摘要模块(Text summary)：   
  文本摘要模块的预处理比较麻烦，步骤比较多。本程序训练Word2vect模型用的是中文维基百科语料库，读者可自行下载https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
  然后，安装Wikipedia Extractor，使用Wikipedia Extractor抽取正文内容。Wikipedia Extractor是意大利人用Python写的一个维基百科抽取器，使用非常方便。下载之后直接使用这条命令即可完成抽取，运行时间很快。执行以下命令。
  $ sudo apt-get install unzip python python-dev python-pip
  $ git clone https://github.com/attardi/wikiextractor.git wikiextractor
  $ cd wikiextractor
  $ sudo python setup.py install
  $ ./WikiExtractor.py -b 1024M -o extracted zhwiki-latest-pages-articles.xml.bz2
  Windows使用powershell也是一样的命令(除去sudo哈..)
  
  



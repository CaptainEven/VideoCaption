# VideoCaption
视频的文本摘要(标注)，输入一段视频，通过深度学习网络和人工智能程序识别视频主要表达的意思
Video summary with text, input a video output a txt decribing the video.
</br>
</br>
## 本程序总共包含3个模块：</br>
(1). 视频读取与关键帧提取模块 </br>
(2). Image caption模块(通过训练一个CNN feature extracter + LSTM网络) </br>
(3). Text summary模块 </br>

依赖包:
-
python3, numpy, opencv, pytorch, jieba分词, textrank4zh, tdqm, opencc, gensim </br>
</br>
预训练的模型文件：
Image caption模块的模型文件和Text summary模型文件，链接：</br>
[模型文件链接](https://pan.baidu.com/s/1thMC-pQ31xgl68ceDuIERg)
</br></br>
使用方法：</br>
-
python videoCaption.py video_file </br>
</br>

脚本运行效果截图：(测试的视频是薛之谦的《演员》mv)-
![](https://github.com/CaptainEven/VideoCaption/blob/master/screen%20shots/result.png)
</br> </br>
算法主要步骤和脚本文件详解：</br>
-
## (1).Short Detector模块:
  通过opencv读入视频流，使用3帧间差法：计算相邻2帧的直方图帧间一阶差分和二阶差分算子,然后根据阈值判断是否切换镜头，保存镜头所在的帧ID即可。
  视频文件较大也可以，程序会将视频文件通过流的方式逐步读入内存。</br>
</br>
## (2). Image Caption模块:
  本模块将深度卷积神经网络和深度循环神经网络结合，用于解决图像标注和语句检索问题。通过CNN提取输入图像的高层语义信息，然后输入到LSTM不断预测下一个最可能出现的词语，组成图像描述。训练的目标就是输出的词语与预期的词语相符合，依次设计神经网络的loss函数。本程序提供训练好的模型，链接见上。读者想要用自己的数据及训练也是可以的。</br>
  通过调用img2txt.py的generate_txt()函数，输入预处理后的图像数据，输出图像描述信息。</br>
  [训练数据来源-全球AI挑战赛-图像中文描述](https://challenger.ai/competition/caption/) </br>
  需要注意的是：</br>
  ### <1>. 这里使用ResNet作为图像高层次语义特征提取模型，不能直接使用ResNet模型，需要做出一点修改：</br>
      模型要去掉网络的最后一层全连接层FC，或者将FC替换为恒等映射。因为FC主要作用是从特征空间映射到样本空间，起到特征融合的作用，为分类做准备。我们这里不需要得到分类概率，只需要特征信息即可。
</br></br>
## (3). 文本摘要模块(Text summary)：
  文本摘要模块使用的是textRank算法：类似于PageRank,不同之处在于将每一个`句子`看作网络中的`节点`。</br>
  在进行textRank之前，将句子处理成一个由词语(word)组成的list。通过计算句子与其他句子之间的关联程度(相似性程度)来计算句子在文中的重要性程度。这里提供两种算法：</br>
  ### <1>. 将句子转化为词频向量，然后计算词频向量之间的夹角余弦作为句子之间的相似度，即textrank4zh的做法。</br>
  textrank4zh计算关键词使用的方法是N-gams + textrank。词语是否相邻决定词语之间是否存在着Edge连接。</br>
  N-gram模型假设在不改变词语在上下文的顺序的前提下，在文中物理距离越近，则关联度越大，物理距离越远则关联度越小，这个假设在某些情况下并不合理，因为没有考虑距离越远的词有可能反而关联度更近的情况。</br>
  ### <2>. 词向量是多维实数向量，向量中包含了自然语言中的语义和语法关系信息。不仅考虑了上下文信息，而且减少了冗余信息，因此可以很自然可以想到利用word2vect计算句子与句子之间的相似性：</br>
词向量均值法：</b> 
  例如计算句子A=['word','you','me']，与句子B=['sentence','google','python']计算相似性，从word2vec模型中分别得到A中三个单词的词向量v1,v2,v3取其平均值Va(avg)=(v1+v2+v3)/3。对句子B做同样的处理得到Vb(avg)，然后计算Va(avg)与Vb(avg)连个向量的夹角余弦值，Cosine Similarity视为句子A与B的相似度值。</br>
### <3>. 文本摘要还可以其他深度学习的方式，比如通过训练一个seq2seq模型来完成，这部分内容还在测试，后续会补上。</br>
关于文本摘要使用的维基百科中文预料处理：</br>
-
  文本摘要模块的预处理比较麻烦，步骤比较多。本程序训练Word2vect模型用的是中文维基百科语料库，读者可自行下载https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
  然后，安装Wikipedia Extractor，使用Wikipedia Extractor抽取正文内容。Wikipedia Extractor是意大利人用Python写的一个维基百科抽取器，使用非常方便。下载之后直接使用这条命令即可完成抽取，运行时间很快。执行以下命令。
  </br>
  $ sudo apt-get install unzip python python-dev python-pip </br>
  $ git clone https://github.com/attardi/wikiextractor.git wikiextractor </br>
  $ cd wikiextractor </br>
  $ sudo python setup.py install </br>
  $ ./WikiExtractor.py -b 1024M -o extracted zhwiki-latest-pages-articles.xml.bz2 </br>
  Windows使用powershell也是一样的命令(注意除去sudo)，命令运行结束会在目录extracted的下一级目录下得到两个文件wiki_00, wiki_01。</br>
 接下里对这两个文件做预处理: </br>
## (1). 繁体转简体: </br>
    使用opencc(windows下安装比较麻烦，最有效的方式直接下载opencc-python绑定包源码，直接通过源码的setup.py安装，使用过程中可能会遇到版本问题，注释掉相应的代码即可，不影响使用，亲测)。 </br>
    linux下直接运行脚本进行opencc的安装和繁转简处理：</br>
     $ sudo apt-get install opencc </br>
     $ opencc -i wiki_00 -o zh_wiki_00 -c zht2zhs.ini </br>
     $ opencc -i wiki_01 -o zh_wiki_01 -c zht2zhs.ini </br>
 ## (2). 去除多余的符号，清洗文本，使用preprocess下的filter_wiki.py脚本。</br>
 ## (3). 使用preprocess下的cut2words.py处理，主要是通过jieba做分词处理。</br>
 ## (4). 安装gensim，通过train_word2vect.py训练word2vect模型，训练结束得到3个模型文件。利用gensim可以计算中文词语的词向量 </br>
 可以通过提供的Word2VectTextRank.py脚本和test_0.txt, test_1.txt..测试文件测试文本摘要效果。
 ![](https://github.com/CaptainEven/VideoCaption/blob/master/screen%20shots/text_sum.png)
     
  



# coding:utf-8
import os
# import cv2
import numpy as np
from main import *
from textrank4zh import *

#-------------------test code
# # 切换到脚本所在目录
# SCRIPT_PATH = os.path.split(os.path.realpath(__file__))[0]  # 取脚本所在目录
# print('script path: ', SCRIPT_PATH)
# os.chdir(SCRIPT_PATH)

# opt = Config()
# opt.caption_data_path = 'caption.pth'
# opt.test_img = 'frames/380.jpg'
# opt.use_gpu = False
# opt.model_ckpt = 'caption_0914_1947'

# # 数据预处理
# data = t.load(opt.caption_data_path)
# word2ix, ix2word = data['word2ix'], data['ix2word']

# IMG_NET_MEAN = [0.485, 0.456, 0.406]
# IMG_NET_STD = [0.229, 0.224, 0.225]
# normalize = tv.transforms.Normalize(mean=IMG_NET_MEAN, std=IMG_NET_STD)
# transforms = tv.transforms.Compose([
#     tv.transforms.Resize(opt.scale_size),
#     tv.transforms.CenterCrop(opt.img_size),
#     tv.transforms.ToTensor(),
#     normalize
# ])

# img_ = Image.open(opt.test_img)
# img_ = img_.convert('RGB') # 转换为3通道的格式(RGB)
# # img_.show()
# img = transforms(img_).unsqueeze(0)
# img_.resize((int(img_.width * 256 / img_.height), 256))


# # img_ = cv2.resize(np.array(img_),
# #            ((int(img_.width * 256 / img_.height), 256)),
# #            interpolation=cv2.INTER_CUBIC)
# # print('width, height: ', img_.shape[1], img_.shape[0])

# # 用resnet50提取图像特征:如果resnet模型文件不存在会自动下载
# resnet50 = tv.models.resnet50(True).eval()
# del resnet50.fc
# resnet50.fc = lambda x: x  # 将全连接层替换为恒等映射
# resnet50.avgpool.stride = 7 # 修改average pool的步长

# if opt.use_gpu:
#     resnet50.cuda()
#     img = img.cuda()
# img_feats = resnet50(Variable(img, volatile=True))

# # 应用Caption模型进行图像描述
# model = CaptionModel(opt, word2ix, ix2word)
# model = model.load(opt.model_ckpt).eval()
# if opt.use_gpu:
#     model.cuda()

# results = model.generate(img_feats.data[0])
# # print('\r\n'.join(results))

# # 拼接results
# text = ''
# for sentence in results:
#     text += sentence[:-6]
#     text += '。'
# print(text, '\n')

# # tr4w = TextRank4Keyword()
# # tr4w.analyze(text=text, lower=True, window=2)

# tr4s = TextRank4Sentence()
# tr4s.analyze(text=text, lower=True, source='all_filters')
# # for item in tr4s.get_key_sentences(num=3):
# #     print(item.index, item.weight, item.sentence)
# img_txt = tr4s.get_key_sentences()[0].sentence + '。'
# print('img summary:\n', img_txt)
# print('--Test done.')
#-------------------test code


# 待改造优化: 模型只加载一次
def generate_txt(img,
                 tr4s,
                 feat_extractor,
                 model,
                 use_gpu):
    '''
    @input image
    @return image caption
    '''
    if use_gpu:
        feat_extractor.cuda()
        img = img.cuda()
    img_feats = feat_extractor(Variable(img, volatile=True))

    # 应用Caption模型进行图像描述
    if use_gpu:
        model.cuda()

    results = model.generate(img_feats.data[0])
    # print('\r\n'.join(results))

    # 拼接results
    text = ''
    for sentence in results:
        text += sentence[:-6]
        text += '。'

    tr4s.analyze(text=text, lower=True, source='all_filters')
    img_txt = tr4s.get_key_sentences()[0].sentence + '。' \
    + tr4s.get_key_sentences()[1].sentence + '。' \
    + tr4s.get_key_sentences()[2].sentence + '。'
    return img_txt

# coding: utf-8
from ShortDetect import get_key_frames
from img2txt import *
import cv2
import numpy as np
from tqdm import tqdm
from Word2VectTextRank import summarize

def video2txt(video_path, save_key_frame=False):
    '''
    @input: 
    '''
    PATH = os.path.split(os.path.realpath(video_path))[0] + '/'
    
    # 生成关键帧
    key_frames, IMG_SIZE = get_key_frames(video_path)

    # 存储关键帧到硬盘
    # if save_key_frame:
    #     print('\n--saving key frames...')
    #     key_frame_path = os.path.join(PATH + 'key_frames')
    #     if not os.path.exists(key_frame_path):
    #         os.makedirs(key_frame_path)
    #     for i, frame in enumerate(key_frames):
    #         kf_path = os.path.join(key_frame_path + '/' + str(i) + '.jpg')
    #         if not os.path.exists(kf_path):
    #             cv2.imwrite(kf_path, frame)
    #             print('--{} saved.'.format(kf_path))

    # 模型加载
    opt = Config()
    opt.caption_data_path = './caption.pth'
    opt.model_ckpt = './caption_0914_1947'
    opt.use_gpu = False

    data = t.load(opt.caption_data_path)
    word2ix, ix2word = data['word2ix'], data['ix2word']

    IMG_NET_MEAN = [0.485, 0.456, 0.406]
    IMG_NET_STD = [0.229, 0.224, 0.225]

    normalize = tv.transforms.Normalize(mean=IMG_NET_MEAN, std=IMG_NET_STD)
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.scale_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        normalize
    ])

    resnet50 = tv.models.resnet50(True).eval()  # 用resnet50提取图像特征
    del resnet50.fc
    resnet50.fc = lambda x: x  # 将全连接层替换为恒等映射
    resnet50.avgpool.stride = 7  # 修改average pool步长

    cap_model = CaptionModel(opt, word2ix, ix2word) # 加载图像描述模型
    cap_model = cap_model.load(opt.model_ckpt).eval()

    # 为每一帧image生成Caption
    tr4s = TextRank4Sentence()
    is_resize = True
    if max(IMG_SIZE) == 256:
        is_resize = False

    print('\n--processing key frames...')
    txts = ''
    for frame in tqdm(key_frames):
        # 处理每帧图像
        frame = Image.fromarray(frame).convert('RGB')  # 转换为3通道的格式(RGB)
        if is_resize:
            frame.resize(IMG_SIZE)
        img = transforms(frame).unsqueeze(0)
        txts += generate_txt(img, tr4s, resnet50, cap_model, opt.use_gpu)
    txts = ''.join(txts.split()) # 去空格
    print('all img_txts:\n', txts)  # 字符串数组

    # ----------------文本摘要(有很多算法,这里尝试两种算法,未尝试的算法如seq2seq)
    # 算法一: textRank
    tr4s.analyze(text=txts, lower=True, source='all_filters')
    summary = tr4s.get_key_sentences()[0].sentence + '。' \
    + tr4s.get_key_sentences()[1].sentence + '。' \
    + tr4s.get_key_sentences()[2].sentence + '。' \
    # + tr4s.get_key_sentences()[3].sentence + '。' \
    # + tr4s.get_key_sentences()[4].sentence + '。' # 取3个最重要的句子
    
    summary = ''.join(summary.split())
    print('video caption 1:\n', summary)

    # 算法二: word2vect based textRank
    summary = summarize(txts, 2) # 排序后,取2个句子
    sum_2 = ''
    for sent in summary:
        sum_2 += sent
    print('video caption 2:\n', sum_2)

    # 保存summary
    # print('--saving text...')
    # txts_path = os.path.join(PATH + 'summary.txt')
    # print('txts_path: ', txts_path)
    # with open(txts_path, "w", encoding='utf-8') as txt_file:
    #     txt_file.write(txts)

    # if (len(key_frames) != 0):
    #     cv2.imshow('key_frame example', key_frames[int(len(key_frames) * 0.5)])
    #     cv2.waitKey()
    # else:
    #     print('[error]: extract key frames failed.')
    #     return


# if __name__ == '__main__':
#     video2txt('./actor.mp4', True)
#     print('--Test done.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python videoCaption.py input_file")
        sys.exit()
    in_file = sys.argv[1]
    print('in_file: ', in_file)
    video2txt(in_file)
    print('--Video caption done.')

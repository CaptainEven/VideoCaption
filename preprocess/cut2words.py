#coding:utf-8
import sys
import os
import jieba
import re
from tqdm import tqdm

def cut_file(in_file):
    if not os.path.exists(in_file):
        print('[error]: invalid file path.')
        return
    PATH = os.path.split(os.path.realpath(in_file))[0] # __file__
    prefix = os.path.split(os.path.realpath(in_file))[1]
    out_file = os.path.join(PATH + '/space_cut_' + prefix)
    print('out_file: ', out_file)
    f_in = open(in_file, 'r', encoding='utf-8') # 以utf-8格式读取
    f_out = open(out_file, 'w+', encoding='utf-8') # 以utf-8格式写
    for line in tqdm(f_in.readlines()):
        words = jieba.cut(line)
        for word in words:
            f_out.write(str(word) + ' ') # 是否应该为分词之间加入空格?
        f_out.write('\n') # 行与行之间是否应该加入换行符?
    f_in.close()
    f_out.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python cut2words.py input_file")
        sys.exit()
    in_file = sys.argv[1]
    print('in_file: ', in_file)
    cut_file(in_file)
    print('--cut done.')


# http://blog.csdn.net/qq_32166627/article/details/68942216

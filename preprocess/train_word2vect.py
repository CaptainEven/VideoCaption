# coding:utf-8
from gensim.models import word2vec
import logging
import os
import sys


def train(in_file):
    '''
    train a word2vect model using processd file
    '''
    if not os.path.exists(in_file):
        print('[error]: invalid file path.')
        return
    PATH = os.path.split(os.path.realpath(in_file))[0] + '/'  # __file__
    prefix = os.path.split(os.path.realpath(in_file))[1]
    out_file = os.path.join(PATH + prefix + '_wiki_ch_model')
    print('out_file: ', out_file)
    
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(in_file)
    model = word2vec.Word2Vec(
        sentences, size=200, window=5, min_count=5, workers=8)
    model.save(out_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python train_word2vect.py in_file')
        sys.exit()
    in_file = sys.argv[1]
    print('in_file: ', in_file)
    train(in_file)
    print('--model trained done.')

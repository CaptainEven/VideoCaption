#coding:utf-8
import os
import sys
from gensim.models import word2vec

def test(model_path, word_1, word_2):
    '''
    test from trained model and input
    '''
    if not os.path.exists(model_path):
        print('[error]: model path is not valid.')
        return
    model = word2vec.Word2Vec.load(model_path)
    print(model.wv.similarity(word_1, word_2))


if __name__ =='__main__':
    if len(sys.argv) != 4:
        print('--Usage: python test_word2vect.py model_path word_1, word_2')
        sys.exit()
    test(sys.argv[1], sys.argv[2], sys.argv[3])
    print('--Test done.')

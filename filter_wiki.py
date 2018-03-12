# -*- coding: utf-8 -*-
import re
import sys
import codecs


def filter(input_file):
    p1 = re.compile('（）')
    p2 = re.compile('《》')
    p3 = re.compile('「')
    p4 = re.compile('」')
    p5 = re.compile('<doc (.*)>')
    p6 = re.compile('</doc>')
    out_file = codecs.open('std_' + input_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = p1.sub('', line)
            line = p2.sub('', line)
            line = p3.sub('', line)
            line = p4.sub('', line)
            line = p5.sub('', line)
            line = p6.sub('', line)
            out_file.write(line)
    out_file.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py inputfile")
        sys.exit()
    reload(sys)
    sys.setdefaultencoding('utf-8')
    input_file = sys.argv[1]
    filter(input_file)

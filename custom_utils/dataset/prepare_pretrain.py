#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/3/1 19:34
# @Author : WeiHua
import json
from tqdm import tqdm


if __name__ == '__main__':
    corpus = []
    with open('/apdcephfs/share_887471/common/whua/dataset/ie_e2e/SynthText/train.txt', 'r', encoding='utf-8') as f:
        for line_ in tqdm(f.readlines()):
            if line_.strip() == "":
                continue
            info_ = json.loads(line_.strip())
            for anno_ in info_['annotations']:
                corpus.append(anno_['text'])
    with open('/apdcephfs/share_887471/common/ocr_benchmark/benchmark/MJSynth/annotation.txt', 'r', encoding='utf-8') as f:
        for line_ in tqdm(f.readlines()):
            if line_.strip() == "":
                continue
            info_ = line_.strip().split(' ')
            if len(info_) != 2:
                print(f"invalid line:{line_}, pass it")
            corpus.append(info_[1])
    with open('/apdcephfs/share_887471/common/whua/st_mj_corpus.txt', 'w', encoding='utf-8') as saver:
        for line_ in tqdm(corpus):
            saver.write(line_+'\n')



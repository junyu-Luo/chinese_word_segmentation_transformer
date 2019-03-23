#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import pkuseg
import logging
import collections
import re


class Tokenizer:
    def __init__(self, logger):
        self.cn_seg = pkuseg.PKUSeg()
        self.en_seg = nltk.tokenize
        self.logger = logger
        self.support_language = ['cn', 'en']

    def tokenize(self, sentence, mode='cn', lower=True):
        '''
        mode:cn 中文
        mode:en 英文
        '''
        if mode not in self.support_language:
            self.logger.error("language mode not supported!")
            exit(1)
        if lower is True:
            sentence = sentence.lower()
        if mode == 'en':
            return self.en_seg.wordpunct_tokenize(sentence)
        elif mode == 'cn':
            sentence = re.sub(r"[ ]+([\u4e00-\u9fa5]+)[ ]*", r"\1", sentence)
            sentence = re.sub(r"[ ]*([\u4e00-\u9fa5]+)[ ]+", r"\1", sentence)
            return self.cn_seg.cut(sentence)
        else:
            self.logger.warn('tokenize sentence "{}" failed!'.format(sentence))
            return []


def get_logger(logger_name, log_file, formatter=None, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    if formatter is None:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter(formatter)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def read_lines(file, mode='r', encoding='utf8', yield_null=False):
    with open(file, mode, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if yield_null:
                yield line
            else:
                if line:
                    yield line


def write_lines(file, data_list, mode='w', encoding='utf8'):
    with open(file, mode, encoding=encoding) as f:
        for line in data_list:
            f.write(line + '\n')


def create_vocabulary(samples, special_symbol=True):
    '''
    统计 vocabulary
    '''
    source_vocab = collections.defaultdict(int)
    target_vocab = set()
    special_vocab = set()
    for sample in samples:
        q, a = sample.split("\t")
        q_s = q.split()
        a_s = a.split()
        if special_symbol:
            for w in q_s:
                if re.search("^<[a-zA-Z]+>$", w):
                    special_vocab.add(w)
                else:
                    source_vocab[w] += 1
            for w in a_s:
                if re.search("^<[a-zA-Z]+>$", w):
                    special_vocab.add(w)
                else:
                    target_vocab.add(w)
        else:
            for w in q_s:
                source_vocab[w] += 1
            for w in a_s:
                target_vocab.add(w)
    return source_vocab, target_vocab, special_vocab


def load_vocabulary(vocab_file):
    id2word = []
    for line in read_lines(vocab_file):
        id2word.append(line)
    word2id = {w: i for i, w in enumerate(id2word)}
    return word2id, id2word

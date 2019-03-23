#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import pickle
import config
import os
import tensorflow as tf
from data_process import Data
from utils import Tokenizer
import logging
import time

Parser = argparse.ArgumentParser()
Parser.add_argument('--model_dir', default='./release', help='release folder')


class test_model:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._pre_process()
        self.load_model()

    def load_model(self):
        graph = tf.Graph()
        with graph.as_default():
            saver = tf.train.import_meta_graph(self.meta_file)
            self.input_ids = graph.get_tensor_by_name(self.var['input_ids'])
            self.decode_ids = graph.get_tensor_by_name(self.var['decode_ids'])
            self.scores = graph.get_tensor_by_name(self.var['scores'])
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=sess_config)
            saver.restore(self.sess, self.model_file)

    def predict_one_sentence(self, sentence):
        sentence_split = self.tokenizer.tokenize(sentence, self.config.source_language_type,
                                                 self.config.source_language_lower)
        s_ids, _ = self.data_tools.encode(sentence_split, 'q')
        feed_dict = {self.input_ids: [s_ids]}
        decode_ids, scores = self.sess.run([self.decode_ids, self.scores], feed_dict)
        decode_ids = decode_ids[0]
        word_lists = []
        for ids in decode_ids:
            word_lists.append(" ".join(self.decode_single(ids)).strip())
        # word_list=self.decode_single(decode_ids)
        # return " ".join(word_list)
        return word_lists

    def test_interactive(self):
        while 1:
            q = input("Q:")
            if q:
                try:
                    t1 = time.time()
                    word_list = self.predict_one_sentence(q)
                    for i, sent in enumerate(word_list):
                        print("A_{}:{}".format(i, sent))
                    t2 = time.time()
                    print("cost time:{}".format(t2 - t1))
                except:
                    break
            else:
                break

    def decode_single(self, decode_ids):
        no_pad_ids = []
        for i, id_ in enumerate(decode_ids):
            if i == 0:
                if id_ == 2:
                    continue
            if id_ == 3:
                break
            no_pad_ids.append(id_)
        word_list = [self.data_tools.id2word_tar[i] for i in no_pad_ids]
        return word_list

    def _pre_process(self):
        self.model_file = os.path.join(self.model_dir, 'model.ckpt')
        self.meta_file = os.path.join(self.model_dir, 'model.ckpt.meta')
        var_file = os.path.join(self.model_dir, 'var.pkl')
        with open(var_file, 'rb') as f:
            self.var, self.config = pickle.load(f)
        basic_config = config.basic_config()
        basic_config.__dict__.update(self.config)
        self.config = basic_config
        source_vocab_file = os.path.join('./data', 'vocab_source.txt')
        target_vocab_file = os.path.join('./data', 'vocab_target.txt')
        self.data_tools = Data(source_vocab_file, target_vocab_file, None, basic_config, logging)
        self.tokenizer = Tokenizer(logging)


if __name__ == "__main__":
    args = Parser.parse_args()
    model = test_model(args.model_dir)
    model.test_interactive()

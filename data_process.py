#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import tensorflow as tf
import os
import glob
import re
from utils import read_lines, write_lines, create_vocabulary, load_vocabulary


def parse_raw_file(source_file, target_file):
    source_list = []
    with open('./data/context.txt', 'r', encoding='utf-8') as f:
        for line in f:
            source_list.append(line.strip())

    target_list = []
    with open('./data/texts.txt', 'r', encoding='utf-8') as f:
        for line in f:
            target_list.append(line.strip())

    assert len(source_list) == len(target_list)
    result = []
    for i in range(len(source_list)):
        result.append([source_list[i], target_list[i]])
    return result


def tokenize_one_line(sentence, cut_fun, specical_symbol, mode, lower):
    raw_sentence = sentence
    tokenized_sentence = []
    if specical_symbol:
        sentence = re.split("(<[a-zA-Z]+>)", sentence)
        for sub_sent in sentence:
            if re.search("^<[a-zA-Z]+>$", sub_sent):
                tokenized_sentence.append(sub_sent)
            else:
                if sub_sent:
                    tokenized_sentence.extend(cut_fun(sub_sent, mode, lower))
    else:
        tokenized_sentence = cut_fun(raw_sentence, mode, lower)
    return cut_white_space(" ".join(tokenized_sentence))


def cut_white_space(sentence):
    return " ".join(sentence.split())


class Data:
    def __init__(self, source_vocab_file, target_vocab_file, sample_file, config, logger):
        self.logger = logger
        self.config = config
        self.sample_file = sample_file
        self.word2id_src, self.id2word_src = load_vocabulary(source_vocab_file)
        self.word2id_tar, self.id2word_tar = load_vocabulary(target_vocab_file)
        self.tf_record_file = os.path.join(self.config.tokenized_data_dir, 'sample.tf_record')
        self.pad_id_src = self.word2id_src['<pad>']
        self.unk_id_src = self.word2id_src['<unk>']
        self.pad_id_tar = self.word2id_tar['<pad>']
        self.unk_id_tar = self.word2id_tar['<unk>']

    def tf_dateset(self):
        self.create_tf_record_file(self.sample_file)
        name_to_features = {
            "q_ids": tf.FixedLenFeature([self.config.max_source_length], tf.int64),
            "a_ids": tf.FixedLenFeature([self.config.max_target_length], tf.int64)}

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)
            return example

        def input_fn():
            """The actual input function."""
            d = tf.data.TFRecordDataset(self.tf_record_file)
            d = d.map(lambda record: _decode_record(record, name_to_features))
            d = d.repeat(self.config.num_epochs)
            d = d.shuffle(buffer_size=10000)
            d = d.batch(self.config.batch_size)
            return d

        # test
        # iterator = d.make_one_shot_iterator()
        # features = iterator.get_next()
        return input_fn

    def create_tf_record_file(self, sample_file):
        '''
        将qa转化为id，并且a添加<s></s>
        '''
        save_file = self.tf_record_file
        if os.path.isfile(save_file):
            self.logger.info('tf record file "{}" existed!'.format(save_file))
            return
        tf_writer = tf.python_io.TFRecordWriter(save_file)
        self.logger.info("Writing example ...")
        num = 0
        for line in read_lines(sample_file):
            q_line, a_line = line.split('\t')
            q_words = q_line.split()
            a_words = a_line.split()
            # if len(q_words)>self.config.max_source_length:
            #    q_words=q_words[:self.config.max_source_length]
            # if len(a_words)>self.config.max_target_length-2:
            #    a_words=a_words[:self.config.max_target_length-2]
            a_words = ['<s>'] + a_words + ['</s>']
            q_ids, q_mask = self.encode(q_words, 'q')
            a_ids, a_mask = self.encode(a_words, 'a')
            # while len(q_ids)<self.config.max_source_length:
            #    q_ids.append(self.pad_id)
            #    q_mask.append(0)
            # while len(a_ids)<self.config.max_target_length:
            #    a_ids.append(self.pad_id)
            #    a_mask.append(0)
            # print(a_words)
            # print(q_words)
            # assert len(q_ids)==self.config.max_source_length
            # assert len(a_ids)==self.config.max_target_length
            features = collections.OrderedDict()
            features["q_ids"] = self.create_int_feature(q_ids)
            features["a_ids"] = self.create_int_feature(a_ids)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            tf_writer.write(tf_example.SerializeToString())
            num += 1
            if num <= 5:
                self.logger.info("*** example {} ***".format(num))
                self.logger.info("source words:{}".format(q_words))
                self.logger.info("source ids:{}".format(q_ids))
                self.logger.info("target words:{}".format(a_words))
                self.logger.info("target ids:{}".format(a_ids))
            if num % 100000 == 0:
                self.logger.info("write sample:{}".format(num))
        self.logger.info("Done! Total examples:{}".format(num))

    def create_int_feature(self, values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    def encode(self, word_list, scope):
        if scope == 'q':
            ids = [self.word2id_src.get(i, self.unk_id_src) for i in word_list]
            mask = [1] * len(ids)
            if self.unk_id_src in ids:
                self.logger.warn("unknown word in {}".format(word_list))
        elif scope == 'a':
            ids = [self.word2id_tar.get(i, self.unk_id_tar) for i in word_list]
            mask = [1] * len(ids)
            if self.unk_id_tar in ids:
                self.logger.warn("unknown word in {}".format(word_list))
        else:
            self.logger.error("Something wrong durting encoding.")
            exit(1)
        return ids, mask

    @property
    def vocab_size_src(self):
        return len(self.word2id_src)

    @property
    def vocab_size_tar(self):
        return len(self.word2id_tar)

    @property
    def eos_id(self):
        return self.word2id_tar['</s>']

    @staticmethod
    def pre_process_data(raw_data, tokenizer, config, logger):
        '''
        raw_data: dir or a specific file
        '''
        source_vocab_file = os.path.join(config.tokenized_data_dir, 'vocab_source.txt')
        target_vocab_file = os.path.join(config.tokenized_data_dir, 'vocab_target.txt')
        sample_file = os.path.join(config.tokenized_data_dir, 'samples.txt')
        if os.path.isfile(source_vocab_file) and os.path.isfile(target_vocab_file) and os.path.isfile(sample_file):
            logger.info("vocab file and sample file already existed!")
            return Data(source_vocab_file, target_vocab_file, sample_file, config, logger)
        else:
            logger.info("Genarate vocabulary and tokenized samples.")
            source_file = 'vocab_source.txt'
            target_file = 'vocab_target.txt'
            samples = set()
            for qa in parse_raw_file(source_file, target_file):
                q = qa[0]
                a = qa[1]
                tokenized_q = tokenize_one_line(
                    sentence=q,
                    cut_fun=tokenizer.tokenize,
                    specical_symbol=config.special_symbol,
                    mode=config.source_language_type,
                    lower=config.source_language_lower)
                tokenized_a = tokenize_one_line(
                    sentence=a,
                    cut_fun=tokenizer.tokenize,
                    specical_symbol=config.special_symbol,
                    mode=config.target_language_type,
                    lower=config.target_language_lower)
                samples.add(tokenized_q + "\t" + tokenized_a)
            logger.info('sample size:{}'.format(len(samples)))
            logger.info("save samples in '{}'".format(sample_file))
            write_lines(sample_file, samples)
            source_vocab, target_vocab, special_vocab = create_vocabulary(samples, config.special_symbol)
            source_vocab = set(list(source_vocab.keys()))
            for s_symbol in config.vocab_remains:
                if s_symbol in source_vocab:
                    source_vocab.discard(s_symbol)
                if s_symbol in target_vocab:
                    target_vocab.discard(s_symbol)
                if s_symbol in special_vocab:
                    special_vocab.discard(s_symbol)
            logger.info(
                "source vocabulary size:{}".format(len(source_vocab) + len(special_vocab) + len(config.vocab_remains)))
            logger.info(
                "target vocabulary size:{}".format(len(target_vocab) + len(special_vocab) + len(config.vocab_remains)))
            # logger.info('vocab size:{}'.format(len(source_vocab)+len(target_vocab)+len(special_vocab)+len(config.vocab_remains)))
            logger.info('save source vocabulary in "{}"'.format(source_vocab_file))
            with open(source_vocab_file, 'w', encoding='utf8') as f:
                for line in config.vocab_remains:
                    f.write(line + '\n')
                for line in special_vocab:
                    f.write(line + '\n')
                for line in source_vocab:
                    f.write(line + '\n')
            logger.info('save source vocabulary in "{}"'.format(target_vocab_file))
            with open(target_vocab_file, 'w', encoding='utf8') as f:
                for line in config.vocab_remains:
                    f.write(line + '\n')
                for line in special_vocab:
                    f.write(line + '\n')
                for line in target_vocab:
                    f.write(line + "\n")
            return Data(source_vocab_file, target_vocab_file, sample_file, config, logger)


def create_train_data(data_dir):
    from utils import Tokenizer, get_logger
    from config import basic_config
    logger = get_logger('log', './log/log.txt')
    t = Tokenizer(logger)
    model = Data.pre_process_data(data_dir, t, basic_config(), logger)
    model.create_tf_record_file(model.sample_file)
    return model
    # data=d()
    # num=0
    # i=data.make_initializable_iterator()
    # while i.get_next():
    #    num+=1
    #    if num%100000==0:
    #        print(num)
    # print(num)

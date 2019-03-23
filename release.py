#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import tensorflow as tf
import argparse
import pickle
import config
from model import transformer
import numpy as np

Parser = argparse.ArgumentParser()
Parser.add_argument('--release_dir', default='./release', help='release folder')
Parser.add_argument('--steps', default='', help='restore steps')
Parser.add_argument('--restore_dir', default='./out', help='restore folder')


def release_model(**kwargs):
    release_dir = kwargs.get("release_dir", './release')
    restore_dir = kwargs.get('restore_dir', './out')
    if not os.path.isdir(release_dir):
        print("Create release dir:{}".format(release_dir))
        os.mkdir(release_dir)
    for file in glob.glob(os.path.join(release_dir, '*')):
        print("Remove previous file:{}".format(file))
        os.remove(file)
    # release后保存的模型文件，参数文件
    release_model_file = os.path.join(release_dir, 'model.ckpt')
    release_var_file = os.path.join(release_dir, 'var.pkl')
    # restore 的文件
    restore_step = kwargs.get('steps')
    if restore_step:
        restore_model_file = os.path.join(restore_dir, 'model.ckpt-{}'.format(restore_step))
    else:
        restore_model_file = tf.train.get_checkpoint_state(restore_dir).model_checkpoint_path
    restore_var_file = os.path.join(restore_dir, 'options.pkl')
    with open(restore_var_file, 'rb') as f:
        options = pickle.load(f)
        basic_config = config.basic_config()
        basic_config.__dict__.update(options)
        basic_config.beam_size = 2
    g = tf.Graph()
    with g.as_default():
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_ids = tf.placeholder(tf.int64, [None, None], name='input_ids')
            with tf.variable_scope('model'):
                model = transformer.Transformer(basic_config, False)
                out_res = model(input_ids, eos_id=basic_config.eos_id)
            top_decoded_ids = out_res['outputs']
            scores = out_res['scores']
            # print(top_decoded_ids.name)
            # print(scores.name)
            saver = tf.train.Saver()
            saver.restore(sess, restore_model_file)
            saver.save(sess, release_model_file)
            _vars = {'input_ids': input_ids.name, 'decode_ids': top_decoded_ids.name, 'scores': scores.name}
            with open(release_var_file, 'wb') as f:
                pickle.dump((_vars, options), f, -1)
            # res=sess.run(top_decoded_ids,{input_ids:np.array([[2,3,4,5]],dtype=np.int32)})
            # print(res)
            # print(res[0].shape)
            # print(res[1]['k'].shape)
            # print(res[1]['w'].shape)
            print("Done!")


if __name__ == "__main__":
    args = Parser.parse_args()
    release_model(release_dir=args.release_dir,
                  steps=args.steps,
                  restore_dir=args.restore_dir)

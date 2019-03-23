#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pickle
import config
import data_process
import create_dataset
from model import transformer, metrics


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        return learning_rate


def get_train_op(loss, config):
    """Generate training operation that updates variables based on loss."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            config.learning_rate, config.hidden_size,
            config.learning_rate_warmup_steps)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=config.optimizer_adam_beta1,
            beta2=config.optimizer_adam_beta2,
            epsilon=config.optimizer_adam_epsilon)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
    return train_op


def create_model(s_ids, t_ids, mode, config):
    eos_id = config.eos_id
    with tf.variable_scope('model'):
        model = transformer.Transformer(config, mode == tf.estimator.ModeKeys.TRAIN)
        logits = model(s_ids, t_ids, eos_id)
        with tf.variable_scope("loss"):
            xentropy, weights = metrics.padded_cross_entropy_loss(
                logits, t_ids, config.label_smoothing, config.vocab_size_tar)
            # Compute the weighted mean of the cross entropy losses
            loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    return loss


def model_fn_builder(config):
    def model_fn(features, labels, mode):
        tf.logging.info("trainable variables... ")
        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = create_model(features, labels, mode, config)
            train_op = get_train_op(loss, config)
            for var in tf.trainable_variables():
                tf.logging.info(var)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        return output_spec

    return model_fn


def train_input_fn(tf_record_file, config):
    def input_fn():
        return create_dataset.bucket_dataset(tf_record_file, config)

    return input_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    options = config.basic_config()

    tf.gfile.MakeDirs(options.model_save_dir)
    tf.gfile.MakeDirs(options.tokenized_data_dir)
    tf.gfile.MakeDirs(options.log_dir)

    data_model = data_process.create_train_data(options.raw_data_dir)
    options.vocab_size_src = data_model.vocab_size_src
    options.vocab_size_tar = data_model.vocab_size_tar
    options.eos_id = data_model.eos_id
    log_steps = 200
    do_train = True

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=options.model_save_dir,
        log_step_count_steps=log_steps,
        session_config=session_config)

    model_fn = model_fn_builder(options)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=options.model_save_dir, config=run_config)
    if do_train:
        option_file = os.path.join(options.model_save_dir, 'options.pkl')
        with open(option_file, 'wb') as f:
            pickle.dump(options.__dict__, f, -1)
        tf.logging.info("*** options ***")
        for key in options.__dict__:
            tf.logging.info("\t{}:{}".format(key, options.__dict__[key]))
        tf_record_file = './data/sample.tf_record'
        estimator.train(input_fn=train_input_fn(tf_record_file, options))


if __name__ == "__main__":
    main()

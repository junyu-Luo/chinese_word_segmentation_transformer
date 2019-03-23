#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def _get_example_length(example):
    """Returns the maximum length between the example inputs and targets."""
    length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
    return length


def _filter_max_length(example, input_max_length, target_max_length):
    """Indicates whether the example's length is lower than the maximum length."""
    return tf.logical_and(tf.size(example[0]) <= input_max_length,
                          tf.size(example[1]) <= target_max_length)


def _parse_example(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    data_fields = {
        "q_ids": tf.VarLenFeature(tf.int64),
        "a_ids": tf.VarLenFeature(tf.int64)
    }
    # 默认会转化为sparse 数组
    parsed = tf.parse_single_example(serialized_example, data_fields)
    # 转化为dense数组
    inputs = tf.sparse_tensor_to_dense(parsed["q_ids"])
    targets = tf.sparse_tensor_to_dense(parsed["a_ids"])
    return inputs, targets


def _create_min_max_boundaries(max_length, min_boundary, boundary_scale):
    '''
    生成边界集合，在边界范围内的数据将会放在一起。
    这样做是为了提升计算效率。
    参数:
        max_length:最大长度
        min_boundary:最小边界长度
        boundary_scale:边界放大系数
    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
            buckets_min = [0, 4, 8, 16, 24]
            buckets_max = [4, 8, 16, 24, 25]
    '''
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length, min_boundary, boundary_scale):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length, min_boundary, boundary_scale)
    # batch_size的定义是每次训练的word数
    # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
    bucket_batch_sizes = [batch_size // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(example_input, example_target):
        # 计算sample处于哪一个bucket空间
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = _get_example_length((example_input, example_target))
        # TODO: investigate whether removing code branching improves performance.
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)

        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

    return dataset.apply(tf.contrib.data.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def bucket_dataset(tf_record_file, config):
    dataset = tf.data.TFRecordDataset(tf_record_file)
    dataset = dataset.map(lambda x: _parse_example(x))
    # 去掉大于最大长度的sample
    dataset = dataset.filter(
        lambda x, y: _filter_max_length((x, y), config.max_source_length, config.max_target_length))
    max_length = max(config.max_source_length, config.max_target_length)
    dataset = _batch_examples(dataset, config.batch_size, max_length, config.min_boundary, config.boundary_scale)
    dataset = dataset.repeat(config.num_epochs)
    dataset = dataset.prefetch(1)
    return dataset

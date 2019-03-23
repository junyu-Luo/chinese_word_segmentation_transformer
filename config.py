#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 基本配置
class basic_config:
    def __init__(self):
        self.raw_data_dir = './raw_data'  # 原始数据存放的文件夹，可有多个文件
        self.tokenized_data_dir = './data'  # 规整化后的数据存放位置
        self.log_dir = './log'
        self.model_save_dir = './out'  # 训练模型保存地址

        # 输入文本与目标文本的语言类型，cn:中文,en:英文
        self.source_language_type = 'cn'
        self.target_language_type = 'cn'
        self.special_symbol = True  # 是否有特殊符号"<[a-zA-Z]+>"作为特殊符号
        self.source_language_lower = True  # 是否将英文统一为小写
        self.target_language_lower = True
        # vocabulary
        self.vocab_remains = ['<pad>', '<unk>', '<s>', '</s>']
        self.max_source_length = 64
        self.max_target_length = 64

        # data
        self.num_epochs = 50
        self.batch_size = 2048  # 每次训练的字符数
        self.boundary_scale = 2
        self.min_boundary = 4

        """Parameters for the base Transformer model."""
        # Model params
        self.initializer_gain = 1.0  # Used in trainable variable initialization.
        self.hidden_size = 512  # Model dimension in the hidden layers.
        self.num_hidden_layers = 6  # Number of layers in the encoder and decoder stacks.
        self.num_heads = 8  # Number of heads to use in multi-headed attention.
        self.filter_size = 2048  # Inner layer dimensionality in the feedforward network.

        # Dropout values (only used when training)
        self.layer_postprocess_dropout = 0.2
        self.attention_dropout = 0.2
        self.relu_dropout = 0.2

        # Training params
        self.label_smoothing = 0.1
        self.learning_rate = 1.0
        self.learning_rate_decay_rate = 1.0
        self.learning_rate_warmup_steps = 16000

        # Optimizer params
        self.optimizer_adam_beta1 = 0.9
        self.optimizer_adam_beta2 = 0.997
        self.optimizer_adam_epsilon = 1e-09

        # Default prediction params
        self.extra_decode_length = 50
        self.beam_size = 4
        self.alpha = 0.6  # used to calculate length normalization in beam search

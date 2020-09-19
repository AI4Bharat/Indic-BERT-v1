#!/usr/bin/env python3

"""
A script to run IGLUE tasks. Also supports cross-lingual tasks
"""


import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fine_tune.cli import main as finetune_main



for lang in langs:
    argvec = [
            '--train_lang', 'pa',
            '--test_lang', 'pa',
            '--task', 'paraphrase-fuzzy',
            '--model_name_or_path', 'xlm-mlm-100-1280',
            # '--model_name_or_path', 'bert-base-multilingual-cased',
            '--config_name', '',
            '--tokenizer_name', '',
            # '--model_name_or_path', '../models/indic-bert',
            # '--config_name', '../models/indic-bert/config.json',
            # '--tokenizer_name', '../models/indic-bert/spiece.model',
            '--data_dir', '../iglue',
            '--output_dir', '../outputs',
            '--max_seq_length', '128',
            '--learning_rate', '2e-5',
            '--num_train_epochs', '3',
            '--train_batch_size', '16',
            '--seed', '2',
            '--n_gpu', '1'
            ]


    finetune_main(argvec)


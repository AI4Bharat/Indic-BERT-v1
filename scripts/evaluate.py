#!/usr/bin/env python3

"""
A script to run IGLUE tasks. Also supports cross-lingual tasks
"""


import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fine_tune.cli import main as finetune_main



argvec = [
        '--train_lang', 'pa',
        '--test_lang', 'pa',
        '--task', 'paraphrase-fuzzy',
        '--model_name_or_path', 'ai4bharat/indic-bert',
        '--config_name', '',
        '--tokenizer_name', '',
        '--data_dir', '../iglue',
        '--output_dir', '../outputs',
        '--max_seq_length', '128',
        '--learning_rate', '2e-5',
        '--num_train_epochs', '3',
        '--train_batch_size', '32',
        '--seed', '2',
        '--n_gpu', '1'
]

finetune_main(argvec)


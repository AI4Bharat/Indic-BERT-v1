#!/usr/bin/env python3

"""
A script to run IGLUE tasks. Also supports cross-lingual tasks
"""

import argparse
import os
import sys
import itertools as it
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fine_tune.cli import main as finetune_main


if len(sys.argv) != 5:
    print('Usage: python3 run_tasks.py <model_name> <tasks> <train_langs> <test_langs>')
    sys.exit()

model_name = sys.argv[1]
tasks = sys.argv[2].split(',')
train_langs = sys.argv[3].split(',')
test_langs = sys.argv[4].split(',')

BASE_DIR = '/home/divkakwani/refactor'
IGLUE_DIR = '{}/iglue'.format(BASE_DIR)
MODELS_DIR = '{}/models'.format(BASE_DIR)
BASE_OUTPUT_DIR = '{}/outputs'.format(BASE_DIR)

CONFIG = ''
TOKENIZER = ''

if model_name == 'mbert':
    BERT_MODEL = 'bert-base-multilingual-cased'
elif model_name == 'xlmr':
    BERT_MODEL = 'roberta'
else:
    BERT_MODEL = '{}/{}'.format(MODELS_DIR, model_name)
    CONFIG = '{}/{}/config.json'.format(MODELS_DIR, model_name)
    TOKENIZER = '{}/{}/spiece.model'.format(MODELS_DIR, model_name)

IGLUE_TASKS = {
    'agc': ['text_classification', 'indicnlp-articles', '$IGLUE_DIR/indicnlp-articles', True],
    'ner': ['token_classification', 'wikiann-ner', '$IGLUE_DIR/wikiann-ner', True],
    'mep': ['masked_lm', 'wiki-cloze', '$IGLUE_DIR/wiki-cloze', False],
    'wstp': ['multiple_choice', 'wiki-section-titles', '$IGLUE_DIR/wiki-section-titles', True],
    'hp': ['multiple_choice', 'indicnlp-articles-headlines', '$IGLUE_DIR/indicnlp-articles', True],
    'xsr': ['XSR', 'xsent_retrieval', 'mann-ki-baat', '$IGLUE_DIR/cvit-mkb', False]
}

ADDN_TASKS = {
    'tydi': ['question_answering', 'tydi', '$IGLUE_DIR/tydi', True]
}

all_params = list(it.product(tasks, train_langs, test_langs)))

for params in all_params:
    print("=============== Parameters =================")
    print(params)
    print("============================================")

    data_dir = os.path.join(IGLUE_DIR, IGLUE_TASKS[params[0]][2])
    output_dir = os.path.join(BASE_OUTPUT_DIR, params[0], '{}-{}'.format(params[1], params[2]), 'model-'.format(BERT_MODEL))

    argvec = [
        '--train_lang', params[1],
        '--test_lang', params[2],
        '--module_name', IGLUE_TASKS[params[0]][0],
        '--dataset',  IGLUE_TASKS[params[0]][1],
        '--model_name_or_path', BERT_MODEL,
        '--config_name', CONFIG,
        '--tokenizer_name', TOKENIZER,
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--max_seq_length', MAX_LE128NGTH,
        '--learning_rate', 2e-5,
        '--num_train_epochs', 3,
        '--train_batch_size', 32,
        '--seed', 2,
        '--labels', '{}/{}/labels.txt'.format(data_dir, params[2])
        '--n_tpu_cores', 8,
        '--do_predict'
    ]

    if IGLUE_TASKS[params[0]][3]:
        argvec.append('--do_train')

    if params[0] = 'ner':
        print('Calling NER preprocessing script')
        subprocess.call(['scripts/ner_preprocess.sh', data_dir, params[1], params[2])

    finetune_main(argvec)
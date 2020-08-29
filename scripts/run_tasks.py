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


ALL_LANGS = ['pa', 'hi', 'gu', 'mr', 'kn', 'ta', 'te', 'ml', 'or', 'as', 'bn']

BASE_DIR = '/home/divkakwani/refactor'
IGLUE_DIR = '{}/iglue'.format(BASE_DIR)
MODELS_DIR = '{}/models'.format(BASE_DIR)
BASE_OUTPUT_DIR = '{}/outputs'.format(BASE_DIR)

IGLUE_TASKS = {
    'agc': ['text_classification', 'indicnlp-articles', '{}/indicnlp-articles'.format(IGLUE_DIR), True],
    'ner': ['token_classification', 'wikiann-ner', '{}/wikiann-ner'.format(IGLUE_DIR), True],
    'mep': ['masked_lm', 'wiki-cloze', '{}/wiki-cloze'.format(IGLUE_DIR), False],
    'wstp': ['multiple_choice', 'wiki-section-titles', '{}/wiki-section-titles'.format(IGLUE_DIR), True],
    'hp': ['multiple_choice', 'indicnlp-articles-headlines', '{}/indicnlp-articles'.format(IGLUE_DIR), True],
    'xsr': ['XSR', 'xsent_retrieval', 'mann-ki-baat', '{}/cvit-mkb'.format(IGLUE_DIR), False]
}

ADDN_TASKS = {
    'tydi': ['question_answering', 'tydi', '$IGLUE_DIR/tydi', True]
}
SENTIMENT_TASKS = {
    'bbc-news-classification': ['text_classification', 'bbc-articles'],
    'iitp-movie-sentiment': ['text_classification',],
    'iitp-product-sentiment': ['text_classification'],
    'soham-article-classification': ['text_classification'],
    'inltk-headline-classification': ['text_classification'],
    'actsa-sentiment': ['text_classification'],
    'midas-discourse': ['text_classification']
}
                                    }

# Note: need to protect with __main__ for multiprocessing to work
def main():
    if len(sys.argv) != 4:
        print('Usage: python3 run_tasks.py <model_name> <tasks> <test_langs>')
        sys.exit()

    model_name = sys.argv[1]
    tasks = sys.argv[2].split(',')
    train_langs = sys.argv[3].split(',')
    test_langs = sys.argv[4].split(',')

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

    all_params = list(it.product(tasks, test_langs))

    for params in all_params:
        print("=============== Parameters =================")
        print(params)
        print("============================================")

        data_dir = os.path.join(IGLUE_DIR, IGLUE_TASKS[params[0]][2])
        output_dir = os.path.join(BASE_OUTPUT_DIR, params[0], '{}-{}'.format(params[1], params[2]), 'model-{}'.format(model_name))
        
        os.makedirs(output_dir, exist_ok=True)
        
        argvec = [
            '--train_lang', params[1],
            '--test_lang', params[1],
            '--module_name', IGLUE_TASKS[params[0]][0],
            '--dataset',  IGLUE_TASKS[params[0]][1],
            '--model_name_or_path', BERT_MODEL,
            '--config_name', CONFIG,
            '--tokenizer_name', TOKENIZER,
            '--data_dir', data_dir,
            '--output_dir', output_dir,
            '--max_seq_length', '128',
            '--learning_rate', '2e-5',
            '--num_train_epochs', '3',
            '--train_batch_size', '32',
            '--seed', '2',
            '--labels', '{}/{}/labels.txt'.format(data_dir, params[1]),
            '--n_tpu_cores', '8',
            '--do_predict'
        ]

        if IGLUE_TASKS[params[0]][3]:
            argvec.append('--do_train')

        if params[0] == 'ner':
            print('Calling NER preprocessing script')
            subprocess.call(['scripts/ner_preprocess.sh', data_dir, params[1], params[2]])

        finetune_main(argvec)


if __name__ == '__main__':
    main()

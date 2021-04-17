"""
Based on https://github.com/huggingface/transformers/issues/80

"""

import json
import argparse
import glob
import sys
import logging
import os
import time
import string
from filelock import FileLock

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModule, create_trainer
from ..data.examples import InputFeatures
from collections import ChainMap
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


class MaskedLM(BaseModule):

    mode = 'language-modeling'
    output_mode = 'classification'
    example_type = 'multiple-choice'

    def __init__(self, hparams):
        super().__init__(hparams)

        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.test_results_fpath = 'test_results'
        if os.path.exists(self.test_results_fpath):
            os.remove(self.test_results_fpath)

    def convert_examples_to_features(self, examples):

        batch_encoding = self.tokenizer(
            [example.question for example in examples],
            max_length=self.hparams['max_seq_length'],
            padding='max_length',
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            candidates = examples[i].endings
            tokens = [self.tokenizer.tokenize(cand) for cand in candidates]
            token_candidates = []

            for toks in tokens:
                if len(toks) == 0:
                    token_candidates.append(self.tokenizer.unk_token)
                else:
                    token_candidates.append(max(toks, key=lambda t: len(t.strip(string.punctuation))))
            candidate_ids = self.tokenizer.convert_tokens_to_ids(token_candidates)

            feature = InputFeatures(**inputs, candidates=candidate_ids, label=examples[i].label)
            features.append(feature)

        return features

    def test_dataloader(self):
        mode = 'test'
        cached_features_file = self._feature_file(mode)
        if os.path.exists(cached_features_file) and not self.hparams['overwrite_cache']:
            features = torch.load(cached_features_file)
        else:
            features = self.load_features(mode)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids or 0 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        all_cands  = torch.tensor([f.candidates for f in features], dtype=torch.long)
        all_answers  = torch.tensor([f.label for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_cands, all_answers),
            batch_size=self.hparams['eval_batch_size'],
        )

    def test_step(self, batch, batch_idx):
        inputs = {'input_ids': batch[0], 'token_type_ids': batch[2],
                  'attention_mask': batch[1]}

        answers = batch[3].detach().cpu().numpy()
        candidates = batch[4].detach().cpu().numpy()

        # get first mask location
        input_ids = batch[0].detach().cpu().numpy()
        mask_ids = (input_ids == self.mask_id).argmax(axis=1)
        mask_ids = torch.from_numpy(mask_ids)

        predictions = self(**inputs)[0]

        i = torch.arange(0, predictions.shape[0], dtype=torch.int64)
        predictions = predictions[i, mask_ids]
        predictions = predictions.detach().cpu().numpy()

        right, wrong = 0, 0

        for i, pred in enumerate(predictions):
            prob = pred[candidates[i]]
            pred_answer = int(np.argmax(prob))
            if answers[i] == pred_answer:
                right += 1
            else:
                wrong += 1

        return {"right": right, "wrong": wrong}

    def test_epoch_end(self, outputs):
        right = sum(output['right'] for output in outputs)
        wrong = sum(output['wrong'] for output in outputs)
        merged = {'right': right, 'wrong': wrong}

        with FileLock(self.test_results_fpath + '.lock'):
            if os.path.exists(self.test_results_fpath):
                with open(self.test_results_fpath, 'rb') as fp:
                    data = pickle.load(fp)
                data = {'right': data['right'] + merged['right'], 'wrong': data['wrong'] + merged['wrong']}
            else:
                data = merged 
            with open(self.test_results_fpath, 'wb') as fp:
                pickle.dump(data, fp)

        return data

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        return parser

    def run_module(self):
        self.eval()
        self.freeze()
        torch.no_grad()

        trainer = create_trainer(self, self.hparams)

        trainer.test(self)
        preds = pickle.load(open(self.test_results_fpath, 'rb'))
        correct, wrong = preds['right'], preds['wrong']
        with open(os.path.join(self.hparams['output_dir'], 'test_results.txt'), 'w') as fp:
            json.dump({'test_acc': correct/(correct + wrong)}, fp)

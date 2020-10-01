import argparse
import glob
import logging
import os
import subprocess

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss

from .base import BaseModule


logger = logging.getLogger(__name__)


class TokenClassification(BaseModule):

    mode = 'token-classification'
    output_mode = 'classification'
    example_type = 'tokens'

    def __init__(self, hyparams):
        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        script_path = os.path.join(os.path.dirname(__file__), '../..', 'scripts/ner_preprocess.sh')
        cmd = f"bash {script_path} {hyparams['data_dir']} {hyparams['train_lang']} "\
              f"{hyparams['test_lang']} {hyparams['model_name_or_path']} {hyparams['max_seq_length']}"
        subprocess.call(cmd, shell=True)
        
        super().__init__(hyparams)

    def _eval_end(self, outputs):
        """Evaluation called for both Val and Test"""
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds = np.concatenate([x['pred'] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x['target'] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            'val_loss': val_loss_mean,
            'precision': precision_score(out_label_list, preds_list),
            'recall': recall_score(out_label_list, preds_list),
            'f1': f1_score(out_label_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret['log'] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        # when stable
        ret, preds, targets = self._eval_end(outputs)
        logs = ret['log']
        return {'val_loss': logs['val_loss'], 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dict required by pl
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/\
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret['log']
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {'avg_test_loss': logs['val_loss'], 'log': logs, 'progress_bar': logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            '--labels',
            default='',
            type=str,
            help='Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.',
        )
        return parser
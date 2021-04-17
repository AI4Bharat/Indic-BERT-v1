"""
"""
import logging
import json
import os
import pickle
import scipy.spatial as sp
from filelock import FileLock

import numpy as np
import torch

from .base import BaseModule, create_trainer


logger = logging.getLogger(__name__)


class XSentRetrieval(BaseModule):

    mode = 'base'
    output_mode = 'classification'
    example_type = 'text'

    def __init__(self, hparams):
        self.test_results_fpath = 'test_results'
        if os.path.exists(self.test_results_fpath):
            os.remove(self.test_results_fpath)

        super().__init__(hparams)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        last_hidden = outputs[0]
        mean_pooled = torch.mean(last_hidden, 1)
        return mean_pooled

    def test_dataloader_en(self):
        test_features = self.load_features('en')
        dataloader = self.make_loader(test_features, self.hparams['eval_batch_size'])
        return dataloader

    def test_dataloader_in(self):
        test_features = self.load_features('in')
        dataloader = self.make_loader(test_features, self.hparams['eval_batch_size'])
        return dataloader

    def test_step(self, batch, batch_idx):
        inputs = {'input_ids': batch[0], 'token_type_ids': batch[2],
                  'attention_mask': batch[1]}
        labels = batch[3].detach().cpu().numpy()
        sentvecs = self(**inputs)
        sentvecs = sentvecs.detach().cpu().numpy()
        sentvecs = np.hstack([labels[:, None], sentvecs])

        return {'sentvecs': sentvecs}

    def test_epoch_end(self, outputs):
        all_sentvecs = np.vstack([x['sentvecs'] for x in outputs])

        with FileLock(self.test_results_fpath + '.lock'):
            if os.path.exists(self.test_results_fpath):
                with open(self.test_results_fpath, 'rb') as fp:
                    data = pickle.load(fp)
                data = np.vstack([data, all_sentvecs])
            else:
                data = all_sentvecs
            with open(self.test_results_fpath, 'wb') as fp:
                pickle.dump(data, fp)

        return {'sentvecs': all_sentvecs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        return parser

    def run_module(self):
        self.eval()
        self.freeze()

        trainer = create_trainer(self, self.hparams)

        trainer.test(self, self.test_dataloader_en())
        sentvecs1 = pickle.load(open(self.test_results_fpath, 'rb'))
        os.remove(self.test_results_fpath)

        trainer.test(self, self.test_dataloader_in())
        sentvecs2 = pickle.load(open(self.test_results_fpath, 'rb'))
        os.remove(self.test_results_fpath)

        sentvecs1 = sentvecs1[sentvecs1[:, 0].argsort()][:, 1:]
        sentvecs2 = sentvecs2[sentvecs2[:, 0].argsort()][:, 1:]

        result_path = os.path.join(self.hparams['output_dir'], 'test_results.txt')
        with open(result_path, 'w') as fp:
            metrics = {'test_acc': precision_at_10(sentvecs1, sentvecs2)}
            json.dump(metrics, fp)


def precision_at_10(sentvecs1, sentvecs2):
    n = sentvecs1.shape[0]

    # mean centering
    sentvecs1 = sentvecs1 - np.mean(sentvecs1, axis=0)
    sentvecs2 = sentvecs2 - np.mean(sentvecs2, axis=0)

    sim = sp.distance.cdist(sentvecs1, sentvecs2, 'cosine')
    actual = np.array(range(n))
    preds = sim.argsort(axis=1)[:, :10]
    matches = np.any(preds == actual[:, None], axis=1)
    return matches.mean()

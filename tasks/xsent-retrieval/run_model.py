"""
"""
import argparse
import glob
import logging
import os
import time
import pickle
import threading
import sys
import scipy.spatial as sp
from filelock import FileLock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..transformer_base import LightningBase, create_trainer, add_generic_args, loader_from_features
from .utils_xsr import MKBProcessor, convert_examples_to_features


logger = logging.getLogger(__name__)


class SentEncodingTransformer(LightningBase):

    mode = "base"
    output_mode = "classification"

    def __init__(self, hparams):

        self.processor = MKBProcessor(hparams.lang)
        self.test_results_fpath = 'test_results'
        if os.path.exists(self.test_results_fpath):
            os.remove(self.test_results_fpath)

        super().__init__(hparams, mode=self.mode)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        last_hidden = outputs[0]
        # last_hidden = last_hidden[:, :32, :]
        mean_pooled = torch.mean(last_hidden, 1)
        return mean_pooled

    def load_features(self, mode):
        if mode == "en":
            examples = self.processor.get_examples_en(self.hparams.data_dir)
        elif mode == "in":
            examples = self.processor.get_examples_in(self.hparams.data_dir)
        else:
            raise "Invalid mode"

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.hparams.max_seq_length,
            label_list=range(20000),
            output_mode=self.output_mode,
        )
        return features

    def test_dataloader_en(self):
        return self.load_dataset("en", 32)

    def test_dataloader_in(self):
        return self.load_dataset("in", 32)

    def test_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "token_type_ids": batch[2],
                  "attention_mask": batch[1]}
        labels = batch[3].detach().cpu().numpy()
        sentvecs = self(**inputs)
        sentvecs = sentvecs.detach().cpu().numpy()
        sentvecs = np.hstack([labels[:, None], sentvecs])

        return {"sentvecs": sentvecs}

    def test_epoch_end(self, outputs):
        all_sentvecs = np.vstack([x["sentvecs"] for x in outputs])

        with FileLock(self.test_results_fpath + '.lock'):
            if os.path.exists(self.test_results_fpath):
                with open(self.test_results_fpath, 'rb') as fp:
                    data = pickle.load(fp)
                data = np.vstack([data, all_sentvecs])
            else:
                data = all_sentvecs
            with open(self.test_results_fpath, 'wb') as fp:
                pickle.dump(data, fp)

        return {"sentvecs": all_sentvecs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        LightningBase.add_model_specific_args(parser, root_dir)
        return parser


def compute_accuracy(sentvecs1, sentvecs2):
    n = sentvecs1.shape[0]

    # mean centering
    sentvecs1 = sentvecs1 - np.mean(sentvecs1, axis=0)
    sentvecs2 = sentvecs2 - np.mean(sentvecs2, axis=0)

    sim = sp.distance.cdist(sentvecs1, sentvecs2, 'cosine')
    actual = np.array(range(n))
    preds = sim.argsort(axis=1)[:, :10]
    matches = np.any(preds == actual[:, None], axis=1)
    return matches.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SentEncodingTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"xsr_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    model = SentEncodingTransformer(args)
    # model.eval()
    # model.freeze()

    trainer = create_trainer(model, args)

    trainer.test(model, model.test_dataloader_en())
    sentvecs1 = pickle.load(open(model.test_results_fpath, 'rb'))

    os.remove(model.test_results_fpath)

    trainer.test(model, model.test_dataloader_in())
    sentvecs2 = pickle.load(open(model.test_results_fpath, 'rb'))

    sentvecs1 = sentvecs1[sentvecs1[:,0].argsort()]
    sentvecs2 = sentvecs2[sentvecs2[:,0].argsort()]

    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as fp:
        metrics = {'test_acc': compute_accuracy(sentvecs1, sentvecs2))
        json.dump(metrics, fp)


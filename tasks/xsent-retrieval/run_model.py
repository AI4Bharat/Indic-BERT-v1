"""
"""
import argparse
import glob
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train

from iglue import compute_metrics
from iglue import convert_examples_to_features 
from iglue import iglue_output_modes
from iglue import iglue_processors as processors


logger = logging.getLogger(__name__)


class SentEncodingTransformer(BaseTransformer):

    mode = "base"

    def __init__(self, hparams):

        self.processor = MKBProcessor(hparams.lang)

        super().__init__(hparams, self.mode)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        last_hidden = outputs[0]
        mean_pooled = torch.mean(last_hidden, 1)
        return mean_pooled

    def test_dataloader1(self):
        pass

    def test_dataloader2(self):
        pass

    def test_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "token_type_ids": batch[2],
                  "attention_mask": batch[1], "labels": batch[3]}

        sentvecs = self(**inputs)
        sentvecs = sentvecs.detach().cpu().numpy()

        return {"sentvecs": sentvecs}

    def test_epoch_end(self, outputs):
        all_sentvecs = torch.stack([x["sentvecs"] for x in outputs]).detach().cpu()
        return {"sentvecs": sentvecs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--lang",
            default=None,
            type=str,
            required=True,
            help="The language we are dealing with",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        return parser


def compute_accuracy(sentvecs1, sentvecs2):

    dist = F.cosine_similarity(sentvecs1, sentvecs2)
    index_sorted = torch.argsort(dist)
    best = index_sorted[0]


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
    trainer = create_trainer(model, args)

    sentvecs1 = trainer.test(model, test_dataloader=model.test_dataloader1)
    sentvecs2 = trainer.test(model, test_dataloader=model.test_dataloader2)

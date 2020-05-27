"""
Based on https://github.com/huggingface/transformers/issues/80

We don't use pytorch lightning for testing here
"""

import argparse
import glob
import logging
import os
import time
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..transformer_base import LightningBase
from ..utils import add_generic_args
from .utils_mep import MEPProcessor, convert_examples_to_features, compute_metrics


logger = logging.getLogger(__name__)


class MEPTransformer(LightningBase):

    task_name = "missing-entity-prediction"
    mode = "language-modeling"
    output_mode = "classification"

    def __init__(self, hparams):
    	self.hparams = hparams
        self.processor = MEPProcessor(hparams.lang)
        self.labels = self.processor.get_labels()
        self.num_labels = len(self.labels)

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def test(self):

	mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

        for example in self.processors.get_test_examples(self.hparams.data_dir):
	    candidates = example['candidates']
	    candidates = [max(self.tokenizer.tokenize(cand), key=lambda t: len(t.strip(string.punctuation)))
		          for cand in candidates]
            candidates_ids = self.tokenizer.convert_tokens_to_ids(candidates)


            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [0] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            language_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            language_model.eval()

            predictions = language_model(tokens_tensor, segments_tensors)
            predictions_candidates = predictions[0, masked_index, candidates_ids]
            answer_idx = torch.argmax(predictions_candidates).item()

            print(f'The most likely word is "{candidates[answer_idx]}".')

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        LightningBase.add_model_specific_args(parser, root_dir)
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
            help="The input data dir",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = TCTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"agc_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    model = MEPTransformer(args)
    print(model.test())


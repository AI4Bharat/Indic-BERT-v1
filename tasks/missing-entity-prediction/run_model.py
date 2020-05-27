"""
Based on https://github.com/huggingface/transformers/issues/80

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

	self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.test_results_fpath = 'test_results'
        if os.path.exists(self.test_results_fpath):
            os.remove(self.test_results_fpath)

    def forward(self, **inputs):
        return self.model(**inputs)

    def load_features(self, mode):
        if mode in ["dev" or "test"]:
            examples = self.processor.get_test_examples(self.hparams.data_dir)
        else:
            raise "Invalid mode"

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.hparams.max_seq_length,
            label_list=self.labels,
            output_mode=self.output_mode,
        )
        return features

    def test_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "token_type_ids": batch[2],
                  "attention_mask": batch[1]}
        labels = batch[3]   # holds example ids

        predictions = self(**inputs)
        predictions = outputs.detach().cpu().numpy()

        # get first mask location
        input_ids = batch[0]
        mask_ids = (input_ids == self.mask_id).argmax(axis=1)

        predictions = predictions[:, mask_ids, :]
        outputs = np.hstack([labels, predictions])

        return {"outputs": outputs}

    def test_epoch_end(self, outputs):
        all_outputs = np.vstack([x["outputs"] for x in outputs])

        with FileLock(self.test_results_fpath + '.lock'):
            if os.path.exists(self.test_results_fpath):
                with open(self.test_results_fpath, 'rb') as fp:
                    data = pickle.load(fp)
                data = np.vstack([data, all_outputs])
            else:
                data = all_sentvecs
            with open(self.test_results_fpath, 'wb') as fp:
                pickle.dump(data, fp)

        return {"outputs": all_outputs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        LightningBase.add_model_specific_args(parser, root_dir)
        return parser

"""
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
"""

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
    preds1 = pickle.load(open(model.test_results_fpath, 'rb'))

    os.remove(model.test_results_fpath)

    trainer.test(model, model.test_dataloader_in())
    preds2 = pickle.load(open(model.test_results_fpath, 'rb'))

    # print('Accuracy: ', compute_accuracy(sentvecs1, sentvecs2))

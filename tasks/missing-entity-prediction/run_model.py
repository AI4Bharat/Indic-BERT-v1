"""
Based on https://github.com/huggingface/transformers/issues/80

"""

import argparse
import glob
import logging
import os
import time
import string
from filelock import FileLock

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..transformer_base import LightningBase, add_generic_args, create_trainer
from .utils_mep import MEPProcessor, convert_examples_to_features


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

        super().__init__(hparams, self.num_labels, self.mode)

        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.test_results_fpath = 'test_results'
        if os.path.exists(self.test_results_fpath):
            os.remove(self.test_results_fpath)
    
        (candidates, answers) = self.processor.get_test_options(self.hparams.data_dir)

        self.answers = answers

        self.candidates_ids = {}

        for i, candidate in candidates.items():
            candidate = [max(self.tokenizer.tokenize(cand), key=lambda t: len(t.strip(string.punctuation)))
                         for cand in candidate]
            candidate_ids = self.tokenizer.convert_tokens_to_ids(candidate)
            self.candidates_ids[i] = candidate_ids

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
        labels = batch[3].detach().cpu().numpy()   # holds example ids

        predictions = self(**inputs)[0]
        predictions = predictions.detach().cpu().numpy()

        # get first mask location
        input_ids = batch[0].detach().cpu().numpy()
        mask_ids = (input_ids == self.mask_id).argmax(axis=1)

        predictions = predictions[np.arange(predictions.shape[0]), mask_ids, :]
        outputs = np.hstack([labels[:, None], predictions])

        pred_answers = {}
        for pred in predictions:
            cands = self.candidates_ids[pred[0]]
            prob = pred[1:]
            prob = prob[cands]
            pred_answers[pred[0]] = int(np.argmax(prob))

        return {"outputs": pred_answers}

    def test_epoch_end(self, outputs):
        # all_outputs = np.vstack([x["outputs"] for x in outputs])

        # with FileLock(self.test_results_fpath + '.lock'):
        #     if os.path.exists(self.test_results_fpath):
        #         with open(self.test_results_fpath, 'rb') as fp:
        #             data = pickle.load(fp)
        #         data = np.vstack([data, all_outputs])
        #     else:
        #         data = all_outputs
        #     with open(self.test_results_fpath, 'wb') as fp:
        #         pickle.dump(data, fp)

        # return {"outputs": all_outputs}

        merged = {**output for output in outputs}

        with FileLock(self.test_results_fpath + '.lock'):
            if os.path.exists(self.test_results_fpath):
                with open(self.test_results_fpath, 'rb') as fp:
                    data = pickle.load(fp)
                data = {**merged, **data} 
            else:
                data = merged 
            with open(self.test_results_fpath, 'wb') as fp:
                pickle.dump(data, fp)

        return {"outputs": merged}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        LightningBase.add_model_specific_args(parser, root_dir)
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = MEPTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"xsr_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    model = MEPTransformer(args)
    model.eval()
    model.freeze()

    trainer = create_trainer(model, args)

    trainer.test(model)
    preds = pickle.load(open(model.test_results_fpath, 'rb'))
    answers = model.answers

    os.remove(model.test_results_fpath)

    correct, wrong = 0, 0
    for i in answers:
        if i in pred_answers:
            if pred_answers[i] == answers[i]:
                correct += 1
            else:
                wrong += 1
        else:
            raise 'Prediction not found'

    print('Accuracy: ', correct/(correct + wrong))


    # print('Accuracy: ', compute_accuracy(sentvecs1, sentvecs2))

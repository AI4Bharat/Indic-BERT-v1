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

from ..transformer_base import LightningBase, add_generic_args, create_trainer
from .utils_mep import MEPProcessor, convert_examples_to_features
from collections import ChainMap
from torch.utils.data import DataLoader, TensorDataset


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
    

    def forward(self, **inputs):
        return self.model(**inputs)

    def load_features(self, mode):
        if mode in ["dev", "test"]:
            (examples, ids, candidates, answers) = self.processor.get_test_examples(self.hparams.data_dir)
        else:
            raise "Invalid mode"
        
        candidates_ids = []
        for i, candidate in enumerate(candidates):
            candidate = [max(self.tokenizer.tokenize(cand), key=lambda t: len(t.strip(string.punctuation)))
                         for cand in candidate]
            candidate_ids = self.tokenizer.convert_tokens_to_ids(candidate)
            candidates_ids.append(candidate_ids)

        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.hparams.max_seq_length,
            label_list=self.labels,
            output_mode=self.output_mode,
        )
        features = list(zip(features, ids, candidates_ids, answers))
        return features

    def test_dataloader(self):
        mode = 'test'
        cached_features_file = self._feature_file(mode)
        if os.path.exists(cached_features_file) and not self.hparams.overwrite_cache:
            features = torch.load(cached_features_file)
        else:
            features = self.load_features(mode)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f[0].input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f[0].attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f[0].token_type_ids or 0 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f[1] for f in features], dtype=torch.long)
        all_cands  = torch.tensor([f[2] for f in features], dtype=torch.long)
        all_answers  = torch.tensor([f[3] for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_cands, all_answers),
            batch_size=self.hparams.eval_batch_size,
        )

    def test_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "token_type_ids": batch[2],
                  "attention_mask": batch[1]}
        labels = batch[3].detach().cpu().numpy()   # holds example ids
        candidates = batch[4].detach().cpu().numpy()
        answers = batch[5].detach().cpu().numpy()

        # get first mask location
        input_ids = batch[0].detach().cpu().numpy()
        mask_ids = (input_ids == self.mask_id).argmax(axis=1)
        mask_ids = torch.from_numpy(mask_ids)

        predictions = self(**inputs)[0]

        i = torch.arange(0, predictions.shape[0], dtype=torch.int64)
        predictions = predictions[i, mask_ids]
        predictions = predictions.detach().cpu().numpy()

        # predictions = predictions.detach().cpu().numpy()
        # print("predictions: ", predictions.shape)


        # print(input_ids.shape)
        # print(mask_ids.shape)

        # predictions = predictions[np.arange(predictions.shape[0]), mask_ids, :]
        # print(predictions.shape)

        right, wrong = 0, 0

        for i, pred in enumerate(predictions):
            prob = pred[candidates[i]]
            pred_answer = int(np.argmax(prob))
            if answers[i] == pred_answer:
                right += 1
            else:
                wrong += 1

        """
        outputs = np.hstack([labels[:, None], predictions])

        candidates_ids = {}

        for i, l in enumerate(labels):
            l = int(l)
            candidates_ids[l] = candidates[i]

        pred_answers = {}
        for pred in outputs:
            cands = candidates_ids[int(pred[0])]
            prob = pred[1:]
            prob = prob[cands]
            pred_answer = int(np.argmax(prob))
        """

        return {"right": right, "wrong": wrong}

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

        # merged = dict(ChainMap(*[output['outputs'] for output in outputs]))
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
    torch.no_grad()

    trainer = create_trainer(model, args)

    trainer.test(model)
    preds = pickle.load(open(model.test_results_fpath, 'rb'))

    """
    (candidates, answers) = model.processor.get_test_options(model.hparams.data_dir)

    os.remove(model.test_results_fpath)

    correct, wrong = 0, 0
    for i in answers:
        if i in preds:
            if preds[i] == answers[i]:
                correct += 1
            else:
                wrong += 1
        else:
            raise 'Prediction not found'

    """
    correct, wrong = preds['right'], preds['wrong']
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as fp:
        json.dump({'test_acc': correct/(correct + wrong)}, fp)

    # print('Accuracy: ', compute_accuracy(sentvecs1, sentvecs2))

"""
Code inspired from the Huggingface's transformer library:
File path: transformers/examples/text-classification/run_pl_glue.py

To handle large documents, we use head-truncation. Check the following
paper for a detailed analysis of text classification techniques using
bert-like models: https://arxiv.org/pdf/1905.05583.pdf
"""

import argparse
import glob
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..transformer_base import LightningBase, create_trainer, add_generic_args
from .utils_tc import AGCProcessor, convert_examples_to_features, compute_metrics


logger = logging.getLogger(__name__)


class TCTransformer(LightningBase):

    task_name = "article-genre-classificaton"
    mode = "sequence-classification"
    output_mode = "classification"

    def __init__(self, hparams):
        self.hparams = hparams
        self.processor = AGCProcessor(hparams.lang)
        self.labels = self.processor.get_labels(hparams.data_dir)
        self.num_labels = len(self.labels)

        super().__init__(hparams, self.num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def load_features(self, mode):
        if mode == "train":
            examples = self.processor.get_train_examples(self.hparams.data_dir)
        elif mode == "dev":
            examples = self.processor.get_dev_examples(self.hparams.data_dir)
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

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "token_type_ids": batch[2],
                  "attention_mask": batch[1], "labels": batch[3]}

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds)

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dic required by pl
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        LightningBase.add_model_specific_args(parser, root_dir)
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

    model = TCTransformer(args)
    trainer = create_trainer(model, args)

    if args.do_train:
       trainer.fit(model)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)

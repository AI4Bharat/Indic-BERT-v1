
import argparse
import logging
import os
import glob
import random
import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..data import load_dataset
from ..data.examples import *


logger = logging.getLogger(__name__)


MODEL_MODES = {
    'base': AutoModel,
    'sequence-classification': AutoModelForSequenceClassification,
    'question-answering': AutoModelForQuestionAnswering,
    'pretraining': AutoModelForPreTraining,
    'token-classification': AutoModelForTokenClassification,
    'language-modeling': AutoModelWithLMHead,
    'multiple-choice': AutoModelForMultipleChoice,
}


def get_model_class(model_type, mode):
    return MODEL_MODES[mode]


def set_seed(hparams):
    random.seed(hparams['seed'])
    np.random.seed(hparams['seed'])
    torch.manual_seed(hparams['seed'])
    if hparams['n_gpu'] > 0:
        torch.cuda.manual_seed_all(hparams['seed'])


class BaseModule(pl.LightningModule):
    """
    The base module has 4 components: config, tokenizer, transformer model,
    and dataset

    Loading of a dataset:
    1. Load instances of a dataset in the form of `Examples`
    2. Convert all examples into features - may require tokenizer
    3. Create a tensor dataset and loader given all the converted features

    """

    def __init__(self, hparams):
        super().__init__()

        hparams['mode'] = self.mode
        hparams['output_mode'] = self.output_mode
        hparams['example_type'] = self.example_type
        hparams['dev_lang'] = hparams['train_lang']
        self.hparams = hparams  # must come after super
        self.dataset = load_dataset(hparams['dataset'], hparams['data_dir'])
        if self.output_mode == 'classification':
            self.labels = self.dataset.get_labels(hparams['train_lang'])

        # setup config object
        config_name = hparams['config_name'] or hparams['model_name_or_path']
        args = {}
        if self.output_mode == 'classification':
            hparams['num_labels'] = len(self.dataset.get_labels(hparams['train_lang']))
            args = {'num_labels': hparams['num_labels']}

        self.config = AutoConfig.from_pretrained(
            config_name,
            **args,
            cache_dir=hparams['cache_dir']
        )

        # setup tokenizer object
        tok_name = hparams['tokenizer_name'] or hparams['model_name_or_path']
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_name,
            config=self.config,
            cache_dir=hparams['cache_dir'],
        )

        # setup transformer model
        model_class = get_model_class(self.config.model_type, hparams['mode'])
        self.model = model_class.from_pretrained(
            hparams['model_name_or_path'],
            config=self.config,
            cache_dir=hparams['cache_dir'],
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def prepare_data(self):
        """Cache feature files on disk for every mode at the onset"""
        modes = self.dataset.modes()
        for mode in modes:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file)\
                    or self.hparams['overwrite_cache']:
                self.load_features(mode)

    def load_features(self, mode):
        """Load examples and convert them into features"""
        if mode in ('train', 'dev', 'test'):
            lang = self.hparams['{}_lang'.format(mode)]
        else:
            lang = self.hparams['test_lang']
        examples = self.dataset.get_examples(lang, mode)

        cached_features_file = self._feature_file(mode)
        if os.path.exists(cached_features_file)\
                and not self.hparams['overwrite_cache']:
            features = torch.load(cached_features_file)
        else:
            features = self.convert_examples_to_features(examples)
            torch.save(features, cached_features_file)

        return features

    def convert_examples_to_features(self, examples):
        if self.hparams['example_type'] == 'multiple-choice':
            features = convert_multiple_choice_examples_to_features(
                examples,
                self.tokenizer,
                max_length=self.hparams['max_seq_length'],
                label_list=self.labels
            )
        elif self.hparams['example_type'] == 'text':
            features = convert_text_examples_to_features(
                examples,
                self.tokenizer,
                max_length=self.hparams['max_seq_length'],
                label_list=self.labels,
                output_mode=self.output_mode,
            )
        elif self.hparams['example_type'] == 'tokens':
            features = convert_tokens_examples_to_features(
                examples,
                self.labels,
                self.hparams['max_seq_length'],
                self.tokenizer,
                cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
                sep_token=self.tokenizer.sep_token,
                sep_token_extra=bool(self.config.model_type in ["roberta"]),
                pad_on_left=bool(self.config.model_type in ["xlnet"]),
                pad_token=self.tokenizer.pad_token_id,
                pad_token_segment_id=self.tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )
        return features

    def make_loader(self, features, batch_size):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids or 0 for f in features], dtype=torch.long)
        # all_candidates = torch.tensor([f.candidates for f in features], dtype=torch.long)
        if self.hparams['output_mode'] == 'classification':
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams['output_mode'] == 'regression':
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            # TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_candidates),
            batch_size=batch_size,
        )

    def train_dataloader(self):
        train_batch_size = self.hparams['train_batch_size']
        train_features = self.load_features('train')
        dataloader = self.make_loader(train_features, train_batch_size)

        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams['n_gpu'])))
            // self.hparams['gradient_accumulation_steps']
            * float(self.hparams['num_train_epochs'])
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams['warmup_steps'], num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        dev_features = self.load_features('dev')
        dataloader = self.make_loader(dev_features, self.hparams['eval_batch_size'])
        return dataloader

    def test_dataloader(self):
        test_features = self.load_features('test')
        dataloader = self.make_loader(test_features, self.hparams['eval_batch_size'])
        return dataloader

    def training_step(self, batch, batch_idx):
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
        if self.config.model_type != 'distilbert':
            inputs['token_type_ids'] = (
                batch[2] if self.config.model_type in ['bert', 'xlnet', 'albert'] else None
            )  # XLM and RoBERTa don't use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]

        tensorboard_logs = {'loss': loss, 'rate': self.lr_scheduler.get_last_lr()[-1]}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[3]}

        # XLM and RoBERTa don't use token_type_ids
        inputs['token_type_ids'] = None
        if self.config.model_type in ['bert', 'xlnet', 'albert']:
            inputs['token_type_ids'] = batch[2]

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()

        return {'val_loss': tmp_eval_loss.detach().cpu(),
                'pred': preds,
                'target': out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _feature_file(self, mode):
        if mode in ('train', 'dev', 'test'):
            lang = self.hparams['{}_lang'.format(mode)]
        else:
            lang = self.hparams['test_lang']
        return os.path.join(
            self.hparams['data_dir'],
            'cached_{}_{}_{}_{}'.format(
                lang,
                mode,
                list(filter(None, self.hparams['model_name_or_path'].split('/'))).pop(),
                str(self.hparams['max_seq_length']),
            ),
        )

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams['weight_decay'],
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams['learning_rate'],
                          eps=self.hparams['adam_epsilon'])
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None):
        if self.trainer.use_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, 'avg_loss', 0.0)
        tqdm_dict = {'loss': '{:.3f}'.format(avg_loss), 'lr': self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def run_module(self):
        trainer = create_trainer(self, self.hparams)
        hparams_copy = copy.deepcopy(self.hparams)

        if self.hparams['do_train']:
            checkpoints = list(sorted(glob.glob(os.path.join(self.hparams['output_dir'], 'checkpointepoch=*.ckpt'), recursive=True)))
            if len(checkpoints) == 0:
                trainer.fit(self)
                checkpoints = list(sorted(glob.glob(os.path.join(self.hparams['output_dir'], 'checkpointepoch=*.ckpt'), recursive=True)))
            self.trained_model = self.load_from_checkpoint(checkpoints[-1])
            self.trained_model.hparams = hparams_copy

        # Optionally, predict on dev set and write to output_dir
        if self.hparams['do_predict']:
            trainer.test(self.trained_model)


# Fixes __temp_weight_ddp_end.ckpt bug
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/1142
class MonkeyPatchedTrainer(pl.Trainer):
    def load_spawn_weights(self, original_model):
        pass


pl.Trainer = MonkeyPatchedTrainer


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        print(trainer.callback_metrics)

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_dir = pl_module.hparams['output_dir']
            test_lang = pl_module.hparams['test_lang']
            output_test_results_file = os.path.join(output_dir, 'test_results_{}.txt'.format(test_lang))
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


def create_trainer(model, hparams):
    # init model
    set_seed(hparams)

    # if os.path.exists(hparams['output_dir']) and os.listdir(hparams['output_dir']) and hparams['do_train']:
    #   raise ValueError('Output directory ({}) already exists and is not empty.'.format(hparams['output_dir']))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=hparams['output_dir'], prefix='checkpoint', monitor='val_loss', mode='min', save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=hparams['gradient_accumulation_steps'],
        gpus=hparams['n_gpu'],
        max_epochs=hparams['num_train_epochs'],
        early_stop_callback=False,
        gradient_clip_val=hparams['max_grad_norm'],
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    if hparams['fp16']:
        train_params['use_amp'] = hparams['fp16']
        train_params['amp_level'] = hparams['fp16_opt_level']

    if hparams['n_tpu_cores'] > 0:
        train_params['tpu_cores'] = hparams['n_tpu_cores']
        train_params['gpus'] = 0

    if hparams['n_gpu'] > 1:
        train_params['distributed_backend'] = 'ddp'

    trainer = pl.Trainer(**train_params)
    return trainer

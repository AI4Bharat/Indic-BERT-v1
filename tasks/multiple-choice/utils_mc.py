# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

import tqdm
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class WSTPProcessor(DataProcessor):
    """Processor for the Wikipedia Section Title Prediction dataset"""

    def __init__(self, lang):
        self.lang = lang

    def get_train_examples(self, data_dir):
        """See base class."""
        filename = '{}/{}-train.csv'.format(self.lang, self.lang)
        return self._create_examples(read_csv(os.path.join(data_dir, filename)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        filename = '{}/{}-test.csv'.format(self.lang, self.lang)
        return self._create_examples(read_csv(os.path.join(data_dir, filename)), "dev")

    def get_labels(self, data_dir):
        """See base class."""
        filename = '{}/{}-train.csv'.format(self.lang, self.lang)
        lines = read_csv(os.path.join(data_dir, filename))
        labels = map(lambda l: l[0], lines)
        labels = list(set(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def read_json(filepath):
    return json.load(open(filepath, encoding='utf-8'))


class HPProcessor(DataProcessor):
    """Processor for the Headline Predction dataset"""

    def __init__(self, lang):
        self.lang = lang

    def get_train_examples(self, data_dir):
        """See base class."""
        filename = '{}/{}-train.json'.format(self.lang, self.lang)
        return self._create_examples(read_json(os.path.join(data_dir, filename)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        filename = '{}/{}-test.json'.format(self.lang, self.lang)
        return self._create_examples(read_json(os.path.join(data_dir, filename)), "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C", "D"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = [
            InputExample(
                example_id=idx,
                question="",
                contexts=[item['content'], item['content'], item['content'], item['content']],
                endings=[item['optionA'], item['optionB'], item['optionC'], item['optionD']],
                label=item['correctOption'],
            )
            for idx, item in enumerate(items)
        ]
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    label_list: List[str],
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


def compute_metrics(preds, labels):
    return {'acc': (preds == labels).mean()}

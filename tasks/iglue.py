""" IGLUE processors and helpers """

import logging
import os
import csv
from enum import Enum
from typing import List, Optional, Union

from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def read_csv(input_file):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter=","))


class AGCProcessor(DataProcessor):
    """Processor for the Article Genre Classification data set"""

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "agc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


iglue_processors = {
    "agc": AGCProcessor,
}

iglue_output_modes = {
    "agc": "classification",
}

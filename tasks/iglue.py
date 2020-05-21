""" IGLUE processors and helpers """

import logging
import os
import csv
from enum import Enum
from typing import List, Optional, Union

from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def read_csv(input_file):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter=","))


class AGCProcessor(DataProcessor):
    """Processor for the Article Genre Classification data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(read_csv(os.path.join(data_dir, "kn-train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(read_csv(os.path.join(data_dir, "kn-test.csv")), "dev")

    def get_labels(self, data_dir):
        """See base class."""
        lines = read_csv(os.path.join(data_dir, "kn-train.csv"))
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


iglue_processors = {
    "agc": AGCProcessor,
}

iglue_output_modes = {
    "agc": "classification",
}

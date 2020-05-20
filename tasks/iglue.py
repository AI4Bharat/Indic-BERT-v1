""" IGLUE processors and helpers """

import logging
import os
from enum import Enum
from typing import List, Optional, Union

from transformers.data.processors.utils import SingleSentenceClassificationProcessor, DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures


def read_csv(cls, input_file, quotechar=None):
    """Reads a comma separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter=",", quotechar=quotechar))

# monkey-patch read tsv method
DataProcessor._read_tsv = read_csv


class AGCProcessor(SingleSentenceClassificationProcessor):
    """Processor for the Article Genre Classification data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_from_csv(os.path.join(data_dir, "kn-train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_from_csv(os.path.join(data_dir, "kn-test.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

# remove
iglue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

iglue_processors = {
    "agc": AGCProcessor,
}

iglue_output_modes = {
    "agc": "classification",
}

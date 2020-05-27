
import json
import csv
import os

from typing import List, Dict, Optional
from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def read_json(filepath):
    return json.load(open(filepath, encoding='utf-8'))


class MEPProcessor(DataProcessor):
    """Processor for Missing Entity Prediction dataset"""

    def __init__(self, lang):
        self.lang = lang

    def get_test_examples(self, data_dir):
        """See base class."""
        filename = '{}.json'.format(self.lang)
        return self._create_examples(read_json(os.path.join(data_dir, filename)), "test")

    def get_labels(self, data_dir):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, item) in enumerate(items):
            guid = "%s-%s" % (set_type, i)
            text_a = item["question"].replace('<MASK>', '[MASK]')
            label = i   # represent the index of example
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

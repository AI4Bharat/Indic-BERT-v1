
import json
import csv
import os

from typing import List, Dict, Optional
from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def read_json(filepath):
    return json.load(open(filepath, encoding='utf-8'))['cloze_data']


class MEPProcessor(DataProcessor):
    """Processor for Missing Entity Prediction dataset"""

    def __init__(self, lang):
        self.lang = lang

    def get_test_examples(self, data_dir):
        """See base class."""
        filename = '{}.json'.format(self.lang, self.lang)
        return self._create_examples(read_json(os.path.join(data_dir, filename)), "test")

    def get_test_options(self, data_dir):
        filename = '{}.json'.format(self.lang, self.lang)
        items = read_json(os.path.join(data_dir, filename))
        candidates, answers = {}, {}
        for i, item in enumerate(items):
            candidates[i] = item['options']
            answers[i] = item['options'].index(item['answer'])
        return (candidates, answers)

    def get_labels(self):
        """See base class."""
        return list(range(4))

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        candidates = []
        answers = []
        ids = []
        for (i, item) in enumerate(items):
            if 0 in [len(option.strip()) for option in item['options']]:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = item["question"].replace('<MASK>', '[MASK]')
            label = 0   # represent the index of example
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            candidates.append(item['options'])
            answers.append(item['options'].index(item['answer']))
            ids.append(i)
        return examples, ids, candidates, answers

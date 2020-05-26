
import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Dict, Optional

from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)


class MKBProcessor:
    """Processor for Man ki Baat dataset"""

    def __init__(self, lang):
        self.lang = lang

    def get_examples_en(self, data_dir):
        """Get examples of English language"""
        filename = '{}/{}-train.csv'.format(self.lang, self.lang)
        return self._create_examples(read_csv(os.path.join(data_dir, filename)), "train")
    
    def get_examples_in(self, data_dir):
        """Get examples of the Indian language"""
        filename = '{}/{}-train.csv'.format(self.lang, self.lang)
        return self._create_examples(read_csv(os.path.join(data_dir, filename)), "train")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


tasks = {
    'mkb': MKBProcessor
}

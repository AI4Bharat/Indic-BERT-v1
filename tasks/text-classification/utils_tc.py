
import csv
import os

from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features


def read_csv(filepath):
    with open(filepath, encoding='utf-8') as fp:
        return list(csv.reader(fp, delimiter=','))


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

 
def compute_metrics(preds, labels):
    return {'acc': (preds == labels).mean()}



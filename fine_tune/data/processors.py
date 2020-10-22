
import csv
import json
import os

from .examples import MultipleChoiceExample, TextExample, TokensExample


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, lang, mode):
        if mode == 'train':
            return self.get_train_examples(lang)
        elif mode == 'dev':
            return self.get_dev_examples(lang)
        elif mode == 'test':
            return self.get_test_examples(lang)

    def modes(self):
        return ['train', 'dev', 'test']

    def get_train_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, lang):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self, lang):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, encoding='utf-8') as fp:
            return list(csv.reader(fp, delimiter=','))

    @classmethod
    def read_json(cls, input_file):
        """Reads a json file file."""
        with open(input_file, encoding='utf-8') as fp:
            return json.load(fp)

    @classmethod
    def readlines(cls, filepath):
        with open(filepath, encoding='utf-8') as fp:
            return fp.readlines()

    @classmethod
    def read_jsonl(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as fp:
            data = fp.readlines()
            data = list(map(lambda l: json.loads(l), data))
        return data


class IndicNLPHeadlines(DataProcessor):
    """Processor for the Headline Predction dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/{}-train.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'train')

    def get_dev_examples(self, lang):
        '''See base class.'''
        fname = '{}/{}-valid.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'dev')

    def get_test_examples(self, lang):
        '''See base class.'''
        fname = '{}/{}-test.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        return ['A', 'B', 'C', 'D']

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = [
            MultipleChoiceExample(
                example_id=idx,
                question='',
                contexts=[item['content'], item['content'], item['content'],
                          item['content']],
                endings=[item['optionA'], item['optionB'], item['optionC'],
                         item['optionD']],
                label=item['correctOption'],
            )
            for idx, item in enumerate(items)
        ]
        return examples


class WikiCloze(DataProcessor):
    """Processor for Wiki Cloze QA dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def modes(self):
        return ['test']

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath)['cloze_data'], 'test')

    def get_labels(self, lang):
        """See base class."""
        return list(range(4))

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, item) in enumerate(items):
            if '' in [option.strip() for option in item['options']]:
                continue
            example = MultipleChoiceExample(
                example_id=i,
                question=item['question'].replace('<MASK>', '[MASK]'),
                contexts=[],
                endings=item['options'],
                label=item['options'].index(item['answer'])
            )
            examples.append(example)
        return examples


class IndicNLPGenre(DataProcessor):
    """Processor for the Article Genre Classification data set"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/{}-train.csv'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/{}-valid.csv'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'dev')

    def get_test_examples(self, lang):
        fname = '{}/{}-test.csv'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        filename = '{}/{}-train.csv'.format(lang, lang)
        lines = self.read_csv(os.path.join(self.data_dir, filename))
        labels = map(lambda l: l[0], lines)
        labels = list(set(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            example = TextExample(
                guid=('%s-%s' % (set_type, i)),
                text_a=line[1],
                label=line[0]
            )
            examples.append(example)
        return examples


class WikiNER(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_examples(self, lang, mode):
        mode = 'valid' if mode == 'dev' else mode
        file_path = os.path.join(self.data_dir, lang, f'{mode}.txt')
        guid_index = 1
        examples = []
        with open(file_path, encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if words:
                        example = TokensExample(
                            guid=f'{mode}-{guid_index}',
                            words=words,
                            labels=labels
                        )
                        examples.append(example)
                    guid_index += 1
                    words = []
                    labels = []
                else:
                    splits = line.split(' ')
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace('\n', ''))
                    else:
                        # Examples could have no label for mode = 'test'
                        labels.append('O')
            if words:
                example = TokensExample(
                    guid=f'{mode}-{guid_index}',
                    words=words,
                    labels=labels
                )
                examples.append(example)
        return examples

    def get_labels(self, lang):
        path = os.path.join(self.data_dir, lang, 'labels.txt')
        with open(path, 'r') as f:
            labels = f.read().splitlines()
        if 'O' not in labels:
            labels = ['O'] + labels
        return labels


class WikiSectionTitles(DataProcessor):
    """Processor for the Wikipedia Section Title Prediction dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/{}-train.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/{}-valid.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'dev')

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}/{}-test.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        return ['titleA', 'titleB', 'titleC', 'titleD']

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = [
            MultipleChoiceExample(
                example_id=idx,
                question='',
                contexts=[item['sectionText'], item['sectionText'],
                          item['sectionText'], item['sectionText']],
                endings=[item['titleA'], item['titleB'], item['titleC'],
                         item['titleD']],
                label=item['correctTitle'],
            )
            for idx, item in enumerate(items)
        ]
        return examples


class ManKiBaat(DataProcessor):
    """Processor for Man ki Baat dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def modes(self):
        return ['en', 'in']

    def get_examples(self, lang, mode):
        if mode == 'en':
            return self.get_examples_en(lang)
        elif mode == 'in':
            return self.get_examples_in(lang)

    def get_examples_en(self, lang):
        """Get examples of English language"""
        fname = 'en-{}/mkb.en'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.readlines(fpath), 'en')

    def get_examples_in(self, lang):
        """Get examples of the Indian language"""
        fname = 'en-{}/mkb.{}'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.readlines(fpath), 'in')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            example = TextExample(
                guid=('%s-%s' % (set_type, i)),
                text_a=line,
                label=i
            )
            examples.append(example)
        return examples

    def get_labels(self, lang):
        # return dummy value greater than number of examples
        return list(range(10000))


class ACTSA(IndicNLPGenre):
    pass


class BBCNews(IndicNLPGenre):

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/{}-test.csv'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'dev')


class INLTKHeadlines(IndicNLPGenre):
    pass


class SohamArticles(IndicNLPGenre):
    pass


class IITPMovies(IndicNLPGenre):
    pass


class IITProducts(IndicNLPGenre):
    pass


class AmritaParaphraseExact(IndicNLPGenre):

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/{}-test.csv'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'dev')

    def get_labels(self, lang):
        """See base class."""
        filename = '{}/{}-train.csv'.format(lang, lang)
        lines = self.read_csv(os.path.join(self.data_dir, filename))
        labels = map(lambda l: l[2], lines)
        labels = list(set(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            example = TextExample(
                guid=('%s-%s' % (set_type, i)),
                text_a=line[0],
                text_b=line[1],
                label=line[2]
            )
            examples.append(example)
        return examples


class AmritaParaphraseFuzzy(AmritaParaphraseExact):
    pass


class MidasDiscourse(DataProcessor):
    """Processor for the Article Genre Classification data set"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/train.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/val.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'dev')

    def get_test_examples(self, lang):
        fname = '{}/test.json'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_json(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        filename = '{}/train.json'.format(lang, lang)
        lines = self.read_json(os.path.join(self.data_dir, filename))
        labels = map(lambda l: l['Discourse Mode'], lines)
        labels = list(set(labels))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            example = TextExample(
                guid=('%s-%s' % (set_type, i)),
                text_a=line['Sentence'],
                label=line['Discourse Mode']
            )
            examples.append(example)
        return examples


class WNLI(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/train.csv'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/dev.csv'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'dev')

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}/dev.csv'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_csv(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(TextExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class COPA(DataProcessor):
    """Processor for the Wikipedia Section Title Prediction dataset"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, lang):
        """See base class."""
        fname = '{}/train.jsonl'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_jsonl(fpath), 'train')

    def get_dev_examples(self, lang):
        """See base class."""
        fname = '{}/val.jsonl'.format(lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_jsonl(fpath), 'dev')

    def get_test_examples(self, lang):
        """See base class."""
        fname = '{}/val.jsonl'.format(lang, lang)
        fpath = os.path.join(self.data_dir, fname)
        return self._create_examples(self.read_jsonl(fpath), 'test')

    def get_labels(self, lang):
        """See base class."""
        return [0, 1]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = [
            MultipleChoiceExample(
                example_id=idx,
                question='',
                contexts=[item['premise'], item['premise']],
                endings=[item['choice1'], item['choice2']],
                label=item['label'],
            )
            for idx, item in enumerate(items)
        ]
        return examples

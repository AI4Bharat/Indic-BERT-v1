
from .processors import *


PROCESSORS_TABLE = {
    'indicnlp-headlines': IndicNLPHeadlines,
    'wiki-cloze': WikiCloze,
    'indicnlp-genre': IndicNLPGenre,
    'wikiann-ner': WikiNER,
    'wiki-section-titles': WikiSectionTitles,
    'mann-ki-baat': ManKiBaat
}


def load_dataset(dataset_name, data_dir):
    return PROCESSORS_TABLE[dataset_name](data_dir)

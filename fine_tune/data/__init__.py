
from .processors import *


PROCESSORS_TABLE = {
    'indicnlp-articles-headlines': IndicNLPHeadlines,
    'wiki-cloze': WikiCloze,
    'indicnlp-articles': IndicNLPGenre,
    'wikiann-ner': WikiNER,
    'wiki-section-titles': WikiSectionTitles,
    'cvit-mkb': ManKiBaat,
    'actsa': ACTSA,
    'bbc-articles': BBCNews,
    'iitp-movie-reviews': IITPMovies,
    'iitp-product-reviews': IITProducts,
    'inltk-headlines': INLTKHeadlines,
    'soham-articles': SohamArticles,
    'midas-discourse': MidasDiscourse,
    'wnli-translated': WNLI,
    'copa-translated': COPA,
    'amrita-paraphrase-exact': AmritaParaphraseExact,
    'amrita-paraphrase-fuzzy': AmritaParaphraseFuzzy
}


def load_dataset(dataset_name, data_dir):
    return PROCESSORS_TABLE[dataset_name](data_dir)

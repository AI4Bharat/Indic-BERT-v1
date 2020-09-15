
from .processors import *


PROCESSORS_TABLE = {
    'indicnlp-headlines': IndicNLPHeadlines,
    'wiki-cloze': WikiCloze,
    'indicnlp-genre': IndicNLPGenre,
    'wikiann-ner': WikiNER,
    'wiki-section-titles': WikiSectionTitles,
    'actsa': ACTSA,
    'bbc-news': BBCNews,
    'iitp-movies': IITPMovies,
    'iitp-products': IITProducts,
    'inltk-headlines': INLTKHeadlines,
    'soham-articles': SohamArticles,
    'midas-discourse': MidasDiscourse,
    'wnli': WNLI,
    'copa': COPA,
    'amrita-paraphrase-exact': AmritaParaphraseExact,
    'amrita-paraphrase-fuzzy': AmritaParaphraseFuzzy
}


def load_dataset(dataset_name, data_dir):
    return PROCESSORS_TABLE[dataset_name](data_dir)

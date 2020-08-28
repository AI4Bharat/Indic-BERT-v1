

from .masked_lm import MaskedLM
from .multiple_choice import MultipleChoice
from .text_classification import TextClassification
from .token_classification import TokenClassification
from .xsent_retrieval import XSentRetrieval


modules = {
    'masked_lm': MaskedLM,
    'multiple_choice': MultipleChoice,
    'text_classification': TextClassification,
    'token_classification': TokenClassification,
    'xsent_retrieval': XSentRetrieval
}


def get_modules(name=None):
    if name:
        return modules[name]
    return modules.values()

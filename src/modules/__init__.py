

from .masked_lm import MaskedLM
from .multiple_choice import MultipleChoice
from .text_classification import TextClassification
from .token_classification import TokenClassification
from .xsent_retrieval import XSentRetrieval


modules = {
    'masked-lm': MaskedLM,
    'multiple-choice': MultipleChoice,
    'text-classification': TextClassification,
    'token-classification': TokenClassification,
    'xsent-retrieval': XSentRetrieval
}


def get_modules(name=None):
    if name:
        return modules[name]
    return modules.values()

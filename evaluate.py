
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel, BertForMaskedLM



def load_objects(*args, **kwargs):
    if args.model == 'mbert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
    elif args.model == 'xlmr':
        tokenizer = BertTokenizer.from_pretrained('xlm-roberta-base')
        model = BertModel.from_pretrained('xlm-roberta-base')
    elif args.model == 'indicbert':
        pass

    return tokenizer, model



tokenizer, model = load_objects(model='mbert')

import argparse
import os
import sys

from .modules import get_modules


# For every dataset, add an entry:
#   [<transformer module>, <do train?>]
ALL_DATASETS = {
    'indicnlp-articles': ['text_classification', True],
    'wikiann-ner': ['token_classification', True],
    'wiki-cloze': ['masked_lm', False],
    'wiki-section-titles': ['multiple_choice', True],
    'indicnlp-articles-headlines': ['multiple_choice', True],
    'cvit-mkb': ['xsent_retrieval', False],
    'bbc-articles': ['text_classification', True],
    'iitp-movie-reviews': ['text_classification', True],
    'iitp-product-reviews': ['text_classification', True],
    'soham-articles': ['text_classification', True],
    'inltk-headlines': ['text_classification', True],
    'actsa': ['text_classification', True],
    'midas-discourse': ['text_classification', True],
    'wnli-translated': ['text_classification', True],
    'copa-translated': ['multiple_choice', True],
    'amrita-paraphrase-exact': ['text_classification', True],
    'amrita-paraphrase-fuzzy': ['text_classification', True],
}


def add_generic_args(parser, root_dir):
    # task-specific args START
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The evaluation dataset to use'
    )

    parser.add_argument(
        '--lang',
        default=None,
        type=str,
        required=True,
        help='ISO code of test language',
    )
    parser.add_argument(
        '--train_lang',
        default=None,
        type=str,
        help='ISO code of train language. If not specified, it is assumed to be the same as the test langauges',
    )
    # task-specific args END

    # model structural parameters START
    parser.add_argument(
        '--model',
        default=None,
        type=str,
        required=True,
        help='Path to pretrained model or model identifier from huggingface.co/models',
    )

    parser.add_argument(
        '--config_name', default='', type=str, help='Pretrained config name or path if not the same as model_name'
    )

    parser.add_argument(
        '--tokenizer_name',
        default='',
        type=str,
        help='Pretrained tokenizer name or path if not the same as model_name',
    )

    parser.add_argument(
        '--max_seq_length',
        default=128,
        type=int,
        help='The maximum total input sequence length after tokenization. Sequences longer '
        'than this will be truncated, sequences shorter will be padded.',
    )
    # model structural parameters END

    # data I/O args START
    parser.add_argument(
        '--iglue_dir',
        default=None,
        type=str,
        required=True,
        help='The input data dir',
    )

    parser.add_argument(
        '--overwrite_cache', action='store_true', help='Overwrite the cached training and evaluation sets'
    )

    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True,
        help='The output directory where the model predictions and checkpoints will be written.',
    )

    parser.add_argument(
        '--cache_dir',
        default=None,
        type=str,
        help='Where do you want to store the pre-trained models downloaded from s3',
    )
    # data I/O args END

    # model training and inference parameters START
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit',
    )

    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help='For fp16: Apex AMP optimization level selected in ["O0", "O1", "O2", and "O3"].'
        'See details at https://nvidia.github.io/apex/amp.html',
    )

    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_tpu_cores', type=int, default=0)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predictions on the test set.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )

    parser.add_argument('--seed', type=int, default=2, help='random seed for initialization')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
    parser.add_argument(
        '--num_train_epochs', default=3, type=int, help='Total number of training epochs to perform.'
    )
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    # model training and inference parameters END


def main(argvec=None):
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    for module in get_modules():
        module.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args(argvec)
    hparams = vars(args)

    # high-level command line parameters
    dataset = hparams['dataset']
    train_lang = hparams.get('train_lang', hparams['lang'])
    test_lang = hparams['lang']
    model = hparams['model']
    iglue_dir = hparams['iglue_dir']

    data_dir = os.path.join(iglue_dir, dataset)
    output_dir = os.path.join(hparams['output_dir'], dataset,
                              'train-{}'.format(train_lang),
                              'model-{}'.format(model.replace('/', '-')))

    hparams['model_name_or_path'] = hparams['model']
    hparams['train_lang'] = train_lang
    hparams['test_lang'] = test_lang
    hparams['data_dir'] = data_dir
    hparams['output_dir'] = output_dir
    hparams['do_train'] = ALL_DATASETS[dataset][1]
    hparams['do_predict'] = True

    if dataset not in ALL_DATASETS:
        print('Unrecognized dataset')
        sys.exit()

    os.makedirs(output_dir, exist_ok=True)

    module_name = ALL_DATASETS[dataset][0]
    module_class = get_modules(module_name)
    module = module_class(hparams)
    module.run_module()

    return module


if __name__ == '__main__':
    main()

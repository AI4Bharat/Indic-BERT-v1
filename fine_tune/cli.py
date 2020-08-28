import argparse
import os

# from .modules import get_modules


def add_generic_args(parser, root_dir):
    # task-specific args START
    parser.add_argument(
        '--module_name',
        type=str,
        required=True,
        help='The transformer module to use to solve the task'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The dataset used for fine-tuning and evaluation'
    )

    parser.add_argument(
        '--train_lang',
        default=None,
        type=str,
        required=True,
        help='ISO code of train language',
    )

    parser.add_argument(
        '--test_lang',
        default=None,
        type=str,
        required=True,
        help='ISO code of test language',
    )
    # task-specific args END

    # model structural parameters START
    parser.add_argument(
        '--model_name_or_path',
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
        '--data_dir',
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

    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
    parser.add_argument(
        '--num_train_epochs', default=3, type=int, help='Total number of training epochs to perform.'
    )
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    # model training and inference parameters START


def main(argvec=None):
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    for module in get_modules():
        module.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    hparams = vars(argvec)

    module_class = get_modules(hparams['module_name'])
    module = module_class(hparams)
    module.run_module()


if __name__ == '__main__':
    main()

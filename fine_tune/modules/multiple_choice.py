import torch
import numpy as np

from .base import BaseModule
from .utils import mean_accuracy


class MultipleChoice(BaseModule):

    mode = 'multiple-choice'
    output_mode = 'classification'
    example_type = 'multiple-choice'

    def __init__(self, hparams):
        super().__init__(hparams)

    def _eval_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs])\
                             .mean().detach().cpu().item()
        preds = np.concatenate([x['pred'] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=1)

        out_label_ids = np.concatenate([x['target'] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{'val_loss': val_loss_mean},
                   **mean_accuracy(preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret['log'] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret['log']
        return {'val_loss': logs['val_loss'], 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dic required by pl
        logs = ret['log']
        # `val_loss` is the key returned by `self._eval_end()`
        # but actually refers to `test_loss`
        return {'avg_test_loss': logs['val_loss'],
                'log': logs, 'progress_bar': logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        return parser
## Advanced Usage

Note that the following sections describe how to use the fine-tuning CLI for advanced purposes. To do this on Colab, simply use the  arguments mentioned here in the `argvec` list in our [Colab notebook](https://colab.research.google.com/github/ai4bharat/indic-bert/blob/master/notebooks/finetuning.ipynb)

#### Using any Huggingface Model

```python
python3 -m fine_tune.cli --model <HF name*> --dataset <dataset name> --lang <iso lang code> --iglue_dir <base path to indic glue dir> --output_dir <output dir>
```

where HF name refers to the Huggingface shortcut name for the model. For the list of all shortcut names, refer the official docs [https://huggingface.co/transformers/pretrained_models.html](https://huggingface.co/transformers/pretrained_models.html)



#### Loading Model from Local File

All models in the code are loaded through HF transformers library. For any model, you need the following three files:

* `config.json`: config file in HF format; check config files used by transformers, for example [here](https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py).
* `tok.model`: the tokenizer  (spm, wordpiece etc.) model file.
* `pytorch_model.bin`: pytorch binary of the transformer model which stores parameters.

If you have tensorflow checkpoints instead of pytorch binary, then use the following command to first generate the pytorch binary file:

```bash
MODEL_DIR=$1

# modify model_type and filenames accordingly
transformers-cli convert --model_type albert \
  --tf_checkpoint $MODEL_DIR/tf_model \
  --config $MODEL_DIR/config.json \
  --pytorch_dump_output $MODEL_DIR/pytorch_model.bin
```

Finally, run the evaluation using the following command:

```bash
python3 -m fine_tune.cli --model <path to the directory containing pytorch_model.bin> --tokenizer_name <path to the tokenizer file> --config_name <path to the config file> --dataset <dataset name> --lang <iso lang code> --iglue_dir <base path to indic glue dir> --output_dir <output dir>
```



#### Running Cross-lingual Experiments

_Add later_
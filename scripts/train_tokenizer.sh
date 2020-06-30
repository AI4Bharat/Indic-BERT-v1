#!/bin/bash

if [ $# != 3 ]; then
    echo "USAGE: ./train_tokenizer.sh <vocab size> <data dir> <output dir>";
    exit
fi

VOCAB_SIZE="$1"
DATA_DIR="$2"
OUTPUT_DIR="$3"
TRAIN_FILE="$DATA_DIR/train_small.txt"

spm_train \
  --input "$TRAIN_FILE"\
  --model_prefix="$OUTPUT_DIR/spm.unigram" --vocab_size="$VOCAB_SIZE" \
  --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
  --control_symbols=[CLS],[SEP],[MASK] \
  --shuffle_input_sentence=true \
  --character_coverage=0.99995 --model_type=unigram

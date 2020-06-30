#!/bin/bash

if [ $# != 3 ]; then
    echo "USAGE: ./train_tokenizer.sh <vocab size> <data dir> <output dir>";
    exit
fi

VOCAB_SIZE="$1"
MTXT_FILE="$2/multilingual.txt"
OUTPUT_DIR="$3"

spm_train \
  --input "$MTXT_FILE"\
  --model_prefix="$OUTPUT_DIR/spm.unigram" --vocab_size="$VOCAB_SIZE" \
  --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
  --control_symbols=[CLS],[SEP],[MASK] \
  --character_coverage=0.99995 --model_type=unigram

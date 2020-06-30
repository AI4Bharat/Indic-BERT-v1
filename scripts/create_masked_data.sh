#!/bin/bash

if [ $# != 2 ]; then
    echo "USAGE: ./create_masked_data.sh <data dir> <output dir>";
    exit
fi

TRAIN_FILE="$1/train.txt"
OUTPUT_DIR="$2"
shards_dir="$OUTPUT_DIR/shards"
data_dir="$OUTPUT_DIR/pretrain"

# create shards
mkdir "$shards_dir"
split --lines=500000 "$TRAIN_FILE" "$shards_dir"

mkdir "$data_dir"

ls "$shards_dir"| xargs -I {} python3 albert/create_pretraining_data.py \
  --input_file="$shards_dir/{}" \
  --output_file="$data_dir/{}.tf_record"  \
  --spm_model_file="$OUTPUT_DIR/spm.unigram.model" \
  --vocab_file="$OUTPUT_DIR/spm.unigram.vocab" \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3

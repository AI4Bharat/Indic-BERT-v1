#!/usr/bin/env bash


DATA_DIR="data3"
LANG="kn"
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased


cat "$DATA_DIR/$LANG/$LANG-train.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/train.txt.tmp"
cat "$DATA_DIR/$LANG/$LANG-valid.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/valid.txt.tmp"
cat "$DATA_DIR/$LANG/$LANG-test.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/test.txt.tmp"


wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"

python3 preprocess.py "$DATA_DIR/$LANG/train.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/train.txt"
python3 preprocess.py "$DATA_DIR/$LANG/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/dev.txt"
python3 preprocess.py "$DATA_DIR/$LANG/test.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/test.txt"

cat "$DATA_DIR/$LANG/train.txt" "$DATA_DIR/$LANG/dev.txt" "$DATA_DIR/$LANG/test.txt" | cut -d " " -f 2 | grep -v "^$"| sort | uniq > "$DATA_DIR/$LANG/labels.txt"

export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=1

export OUTPUT_DIR_NAME=germeval-model
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 run_pl_ner.py --data_dir ./ \
--model_type bert \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict

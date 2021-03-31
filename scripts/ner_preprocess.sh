#!/usr/bin/env bash

export DATA_DIR=$1
export TRAIN_LANG=$2
export TEST_LANG=$3
export BERT_MODEL=$4
export MAX_LENGTH=$5
export SCRIPT="$(dirname $0)/preprocess.py"

cat "$DATA_DIR/$TRAIN_LANG/$TRAIN_LANG-train.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$TRAIN_LANG/train.txt.tmp"
cat "$DATA_DIR/$TRAIN_LANG/$TRAIN_LANG-valid.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$TRAIN_LANG/valid.txt.tmp"
cat "$DATA_DIR/$TEST_LANG/$TEST_LANG-test.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$TEST_LANG/test.txt.tmp"

python3 scripts/preprocess.py "$DATA_DIR/$TRAIN_LANG/train.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$TRAIN_LANG/train.txt"
python3 scripts/preprocess.py "$DATA_DIR/$TRAIN_LANG/valid.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$TRAIN_LANG/valid.txt"
python3 scripts/preprocess.py "$DATA_DIR/$TEST_LANG/test.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$TEST_LANG/test.txt"

cat "$DATA_DIR/$TRAIN_LANG/train.txt" "$DATA_DIR/$TRAIN_LANG/valid.txt" | cut -d " " -f 2 | grep -v "^$"| sort | uniq > "$DATA_DIR/$TRAIN_LANG/labels.txt"

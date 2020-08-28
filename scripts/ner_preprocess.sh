#!/usr/bin/env bash

export DATA_DIR=$1
export TRAIN_LANG=$2
export LANG=$3

cat "$DATA_DIR/$TRAIN_LANG/$TRAIN_LANG-train.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/train.txt.tmp"
cat "$DATA_DIR/$LANG/$LANG-valid.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/valid.txt.tmp"
cat "$DATA_DIR/$LANG/$LANG-test.txt" | awk -F" " '{if($NF>0) {print $1, $(NF)} else {print $0;}}' > "$DATA_DIR/$LANG/test.txt.tmp"

python3 scripts/preprocess.py "$DATA_DIR/$LANG/train.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/train.txt"
python3 scripts/preprocess.py "$DATA_DIR/$LANG/valid.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/valid.txt"
python3 scripts/preprocess.py "$DATA_DIR/$LANG/test.txt.tmp" $BERT_MODEL $MAX_LENGTH > "$DATA_DIR/$LANG/test.txt"

cat "$DATA_DIR/$LANG/train.txt" "$DATA_DIR/$LANG/valid.txt" "$DATA_DIR/$LANG/test.txt" | cut -d " " -f 2 | grep -v "^$"| sort | uniq > "$DATA_DIR/$LANG/labels.txt"
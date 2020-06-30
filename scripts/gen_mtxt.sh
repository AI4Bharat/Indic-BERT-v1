#!/bin/bash

if [ $# != 2 ]; then
    echo "USAGE: ./gen_mtxt.sh <data dir>";
    exit
fi

declare -a langs=("as" "or" "kn" "ml" "ta" "te" "gu" "mr" "en" "hi" "pa" "bn")
DATA_DIR="$1"

# Generate train small file

OUTPUT="$DATA_DIR/train_small.txt"

if [ -f "$OUTPUT" ]; then
    echo "Output file already exists. Please remove it first"
    exit
fi

for lang in ${langs[@]}; do
	echo "Processing $lang"
	lines=$(wc -l "$DATA_DIR/$lang.txt" | cut -d' ' -f1)
	smtlines=$(echo "e(l($lines*100)*0.7)/1" | bc -l)
	smtlines=${smtlines%.*}
	echo "Sampling $smtlines from $lines lines";
	cat "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt"\
        "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt" | head -n "$smtlines" >> "$OUTPUT"
done


# Generate train file
OUTPUT="$DATA_DIR/train.txt"

if [ -f "$OUTPUT" ]; then
    echo "Output file already exists. Please remove it first"
    exit
fi

for lang in ${langs[@]}; do
	echo "Processing $lang"
	lines=$(wc -l "$DATA_DIR/$lang.txt" | cut -d' ' -f1)
	smtlines=$(echo "e(l($lines*2100)*0.7)/1" | bc -l)
	smtlines=${smtlines%.*}
	echo "Sampling $smtlines from $lines lines";
	cat "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt"\
        "$DATA_DIR/$lang.txt" "$DATA_DIR/$lang.txt" | head -n "$smtlines" >> "$OUTPUT"
done


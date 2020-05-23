

export TASK=agc
export DATA_DIR=tasks/data
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=bert-base-multilingual-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/tasks/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 tasks/txtcls.py --data_dir $DATA_DIR \
--task $TASK \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict
# --n_tpu_cores 8

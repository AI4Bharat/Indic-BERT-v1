
lang="kn"

echo "Processing $lang"
export LANG=$lang
export TASK=mep
export DATA_DIR=/home/divkakwani/refactor/wiki-cloze
export MAX_LENGTH=128
export LEARNING_RATE=2e-5
# export BERT_MODEL=bert-base-multilingual-cased # "models/albert-base-orig-full-final/model.ckpt.index"
# export CONFIG=models/albert-base-orig-full-final/albert_base_config.json
# export TOKENIZER=models/albert-base-orig-full-final/spm.unigram.200k.model
export BERT_MODEL="/home/divkakwani/indic-bert/models/albert-base-orig-full-final"
export CONFIG="/home/divkakwani/indic-bert/models/albert-base-orig-full-final/config.json"
export TOKENIZER="/home/divkakwani/indic-bert/models/albert-base-orig-full-final/spiece.model"
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=$LANG-mbert
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/outputs/$TASK/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python3 -m src.cli --data_dir $DATA_DIR \
--train_lang $LANG \
--test_lang $LANG \
--module_name masked-lm \
--dataset wiki-cloze \
--model_name_or_path $BERT_MODEL \
--config_name $CONFIG \
--tokenizer_name $TOKENIZER \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--n_tpu_cores 8 \
--do_train \
--do_predict
# --config_name tasks/albert-base-orig-full-final/albert_base_config.json \
# --tokenizer_name tasks/albert-base-orig-full-final/spm.unigram.200k.model \

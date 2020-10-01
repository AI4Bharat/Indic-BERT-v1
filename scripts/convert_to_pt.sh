
export ALBERT_BASE_DIR=$1

transformers-cli convert --model_type albert \
  --tf_checkpoint $ALBERT_BASE_DIR/tf_model \
  --config $ALBERT_BASE_DIR/config.json \
  --pytorch_dump_output $ALBERT_BASE_DIR/pytorch_model.bin

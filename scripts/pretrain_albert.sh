
python3 -m albert.run_pretraining \
    --input_file=train.tf_record \
    --output_dir=output \
    --albert_config_file=albert_config.json \
    --do_train \
    --train_batch_size=4096 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=125000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000

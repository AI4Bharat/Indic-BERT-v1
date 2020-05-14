
python3 -m albert.run_pretraining \
    --input_file=${STORAGE_BUCKET}/indicnlp-datasets/multilingual/orig/small_tfrecords/*\
    --output_dir=${STORAGE_BUCKET}/albert-base-orig \
    --albert_config_file=configs/albert_base_config.json \
    --do_train \
    --train_batch_size=4096 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=125000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000 \
    --use_tpu \
    --tpu_name=node-2 \
    --tpu_zone=europe-west4-a \
    --num_tpu_cores=8

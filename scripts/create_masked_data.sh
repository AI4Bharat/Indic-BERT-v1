
ls shards | xargs -I {} python3 albert/create_pretraining_data.py \
  --input_file=shards/{} \
  --output_file=pretrain_data/{}.tf_record  \
  --spm_model_file=indicnlp-tokenizer/multilingual/translit1/spm.unigram.100k.model \
  --vocab_file=indicnlp-tokenizer/multilingual/translit1/spm.unigram.100k.vocab \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=3


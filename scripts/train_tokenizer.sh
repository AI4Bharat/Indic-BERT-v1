
spm_train \
  --input train.txt \
  --model_prefix=mult.spm.unigram.200k --vocab_size=200000 \
  --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
  --control_symbols=[CLS],[SEP],[MASK] \
  --character_coverage=0.99995 --model_type=unigram

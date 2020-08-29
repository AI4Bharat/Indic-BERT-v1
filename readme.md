## IGLUE

The repository contains code for running fine-tuning experiments on Indian languages using indic-bert and other bert-like models. We support evaluation on all the datasets that are part of IGLUE as well as other important public datasets.



#### Setting Up

Requires access to a TPU or GPU. RAM of about 30 GB.

Install the dependencies:

```bash
sudo pip3 install -r requirements.txt
```

Set the following environment variables:

```bash
export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models:<path to this repo"
export PYTHONIOENCODING=utf-8
export TPU_IP_ADDRESS="<TPU Internal Address"
export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export LD_LIBRARY_PATH="/usr/local/lib"

```

Install `pytorch xla`:

```bash
curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
sudo python3 pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

```



#### Running Experiments

* TODO: Add how to download models

```
python3 scripts/run_tasks --model_name indic-bert --task agc --langs mr,ta,te
```



#### Try it Out

* Sentiment Classification: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)





#### Supported Tasks

The following IGLUE tasks:

1. Article Genre Classification (AGC)
2. Named Entity Recognition (NER)
3. Headline Prediction (HP)
4. Wikipedia Section Title Prediction (WSTP)
5. Cloze-style Question Answering (WCQA)
6. Cross-lingual Sentence Retrieval (XSR)

Additional IGLUE Tasks:

1. WNLI 
2. COPA

Public Tasks:

1. TyDiQA-SelectP
2. TyDiQA-MinSpan
3. TyDiQA-GoldP
4. Amrita Paraphrase
5. MMQA
6. MIDAS Discourse
7. BBC News Articles
8. IITP Movie Reviews Sentiment Analysis
9. IITP Product Reviews
10. Soham Articles Genre Classification
11. iNLTK Headlines Genre Classifcation
12. ACTSA Sentiment Classifcation




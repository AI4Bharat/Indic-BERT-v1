<div align="center">
    <h1><b><i>Indic BERT</i></b></h1>
  <a href="http://indicnlp.ai4bharat.org">Website</a> |
  <a href="#">Downloads</a> |
    <a href="#">Paper</a><br>
  <img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fgoogle.com&label=Huggingface&color=green&message=indic-bert&logo=huggingface">
  <br><br>
</div>

Indic bert is a multilingual ALBERT model that exclusively covers 12 major Indian languages. It is pre-trained on our novel corpus of around 9 billion tokens and evaluated on a set of diverse tasks. Indic-bert has around 10x fewer parameters than other popular publicly available multilingual models while it also achieves a performance on-par or better than these models.

We also introduce IGLUE - a set of standard evaluation tasks that can be used to measure the NLU performance of monolingual and multilingual models on Indian languages. Alongwith IGLUE, we also compile a list of additional evaluation tasks.  This repository contains code for running all these evaluation tasks on indic-bert and other bert-like models.



### Table of Contents

* [Introduction](#introduction)
* [Setting up the Code]()
* [Running Experiments]()
* [Pretraining Corpus]()
* [IGLUE](#iglue)
* [Additional Evaluation Tasks]()
* [Comparision with Other Models]()
* [Downloads]()
* [Citing]()
* [License]()
* [Contributors]()
* [Contact]()



### Introduction

The Indic BERT model is based on the ALBERT model, a recent derivative of BERT. It is pre-trained on 12 Indian languages: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu.

The easiest way to use Indic BERT is through the Huggingface transformers library. It can be simply loaded like this:

```python
from transformers import AlbertModel, AlbertTokenizer

tokenizer = AlbertTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AlbertModel.from_pretrained('ai4bharat/indic-bert')
```



### Setting up Code

The code can be run on GPU, TPU or on Google's Colab platform. If you want to run it on Colab, you can simply use our fine-tuning notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4bharat/indic-bert/blob/master/notebooks/finetuning.ipynb). For running it in your own VM, start with running the following commands:

```bash
git clone https://github.com/AI4Bharat/indic-bert
cd indic-bert
sudo pip3 install -r requirements.txt
```

By default, the installation will use GPU. For TPU support, first update your `.bashrc` with the following variables:

```bash
export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models:<path to this repo"
export PYTHONIOENCODING=utf-8
export TPU_IP_ADDRESS="<TPU Internal Address"
export TPU_NAME="grpc://$TPU_IP_ADDRESS:8470"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export LD_LIBRARY_PATH="/usr/local/lib"
```

Then, install `pytorch-xla`:

```bash
curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
sudo python3 pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
```



### Running Experiments

To get help, simply run:

```bash
python3 scripts/run_tasks.py --help
```

To evaluate a specific model, execute:

```bash
python3 scripts/run_tasks.py --model <model name> --tasks <comma-separated list of tasks> --langs <comma-separated list of languages>
```



### Pretraining Corpus

We pre-trained indic-bert on AI4Bharat's monolingual corpus. The corpus has the following distribution of languages:


| Language          | as     | bn     | en     | gu     | hi     | kn     |         |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| **No. of Tokens** | 36.9M  | 815M   | 1.34B  | 724M   | 1.84B  | 712M   |         |
| **Language**      | **ml** | **mr** | **or** | **pa** | **ta** | **te** | **all** |
| **No. of Tokens** | 767M   | 560M   | 104M   | 814M   | 549M   | 671M   | 8.9B    |



### IGLUE

This code can be used to evaluate your models on IGLUE, a standardized set of tasks for Indian languages that we propose. It consists of the following tasks:

1. Article Genre Classification (AGC)
2. Named Entity Recognition (NER)
3. Headline Prediction (HP)
4. Wikipedia Section Title Prediction (WSTP)
5. Cloze-style Question Answering (WCQA)
6. Cross-lingual Sentence Retrieval (XSR)



### Additional Evaluation Tasks

##### Natural Language Inference

- Winnograd Natural Language Inference (WNLI)
- Choice of Plausible Alternatives (COPA)

##### Sentiment Analysis

- IITP Movie Reviews Sentiment 
- IITP Product Reviews
- ACTSA Sentiment Classifcation

##### Genre Classification

- Soham Articles Genre Classification
- iNLTK Headlines Genre Classifcation
- BBC News Articles

##### Discourse Analysis

* MIDAS Discourse

##### Question Answering

- TyDiQA-SelectP
- TyDiQA-MinSpan
- TyDiQA-GoldP
- MMQA



### Evaluation Results







### Downloads





### Citing

If you are using any of the resources, please cite the following article:

```
@article{kunchukuttan2020indicnlpcorpus,
    title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085},
}
```

We would like to hear from you if:

- You are using our resources. Please let us know how you are putting these resources to use.
- You have any feedback on these resources.



### License

[![Creative Commons License](https://camo.githubusercontent.com/6887feb0136db5156c4f4146e3dd2681d06d9c75/68747470733a2f2f692e6372656174697665636f6d6d6f6e732e6f72672f6c2f62792d6e632d73612f342e302f38387833312e706e67)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
IndicNLP Corpus  is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).



### Contributors

- Divyanshu Kakwani
- Anoop Kunchukuttan
- Gokul NC
- Satish Golla
- Avik Bhattacharyya
- Mitesh Khapra
- Pratyush Kumar

This work is the outcome of a volunteer effort as part of [AI4Bharat initiative](https://ai4bharat.org).



### Contact

- Anoop Kunchukuttan ([anoop.kunchukuttan@gmail.com](mailto:anoop.kunchukuttan@gmail.com))
- Mitesh Khapra ([miteshk@cse.iitm.ac.in](mailto:miteshk@cse.iitm.ac.in))
- Pratyush Kumar ([pratyushk@cse.iitm.ac.in](mailto:pratyushk@cse.iitm.ac.in))
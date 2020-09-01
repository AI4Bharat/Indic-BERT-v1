<div align="center">
  <h1>Indic BERT</h1>
  <a href="#">website</a> |
  <a href="#">downloads</a> |
  <a href="#">demo</a><br>
  <img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fgoogle.com&label=label&color=green&message=indic-bert&logo=huggingface">
  <br>
</div>

The repository contains code for running experiments for Indian languages on indic-bert and other bert-like models.



### Overview

This code can be used to evaluate your models on IGLUE, a standardized set of tasks for Indian languages that we propose. It consists of the following tasks:

1. Article Genre Classification (AGC)
2. Named Entity Recognition (NER)
3. Headline Prediction (HP)
4. Wikipedia Section Title Prediction (WSTP)
5. Cloze-style Question Answering (WCQA)
6. Cross-lingual Sentence Retrieval (XSR)



### Running Experiments on Colab

- Open finetuning notebook on Colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai4bharat/indic-bert/blob/master/notebooks/finetuning.ipynb)




### Running Experiments Locally

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

Finally, run a task:

```
python3 scripts/run_tasks --model_name indic-bert --task agc --langs mr,ta,te
```




### Additional Tasks


##### NLI Tasks

* WNLI 
* COPA

##### Text Classification Tasks

* MIDAS Discourse
* BBC News Articles
* IITP Movie Reviews Sentiment Analysis
* IITP Product Reviews
* Soham Articles Genre Classification
* iNLTK Headlines Genre Classifcation
* ACTSA Sentiment Classifcation


##### QA Tasks

* TyDiQA-SelectP
* TyDiQA-MinSpan
* TyDiQA-GoldP
* MMQA



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

- Anoop Kunchukuttan
- Divyanshu Kakwani
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
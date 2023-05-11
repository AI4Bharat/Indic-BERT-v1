### As of May 2023, we recommend using [IndicBERT](https://github.com/AI4Bharat/IndicBERT) Repository:
[IndicBERT](https://github.com/AI4Bharat/IndicBERT) is the new and improved implementation of BERT supporting fine-tuning with HuggingFace.
All the download links for IndicCorpv2, IndicXTREME and various IndicBERTv2 models are available [here](https://github.com/AI4Bharat/IndicBERT).

<div align="center">
	<h1><b><i>IndicBERT</i></b></h1>
	<a href="http://indicnlp.ai4bharat.org">Website</a> |
	<a href="#downloads">Downloads</a> |
	<a href="https://indicnlp.ai4bharat.org/papers/arxiv2020_indicnlp_corpus.pdf">Paper</a><br><br>
    <a href="https://huggingface.co/ai4bharat/indic-bert"><img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fhuggingface.co%2Fai4bharat%2Findic-bert&label=Huggingface&color=green&message=indic-bert&logo=huggingface"/></a>&nbsp;<a href="https://github.com/AI4Bharat/indic-bert/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>&nbsp;<a href="https://github.com/AI4Bharat/indic-bert/wiki"><img src="https://img.shields.io/badge/wiki-documentation-forestgreen"/></a>&nbsp;<a href="https://github.com/AI4Bharat/indic-bert/discussions"><img src="https://img.shields.io/badge/forum-discussions-violet"/></a>&nbsp;<a href="https://join.slack.com/t/ai4bharatgroup/shared_invite/zt-jyu2xuoy-djIH1_NM2Gqkm42DgFRXcA"><img src="https://img.shields.io/badge/slack-chat-green.svg?logo=slack"/></a>
  <br><br>
</div>


Indic bert is a multilingual ALBERT model that exclusively covers 12 major Indian languages. It is pre-trained on our novel corpus of around 9 billion tokens and evaluated on a set of diverse tasks. Indic-bert has around 10x fewer parameters than other popular publicly available multilingual models while it also achieves a performance on-par or better than these models.

We also introduce IndicGLUE - a set of standard evaluation tasks that can be used to measure the NLU performance of monolingual and multilingual models on Indian languages. Along with IndicGLUE, we also compile a list of additional evaluation tasks.  This repository contains code for running all these evaluation tasks on indic-bert and other bert-like models.



### Table of Contents

* [Introduction](#introduction)
* [Setting up the Code](#setting-up-the-code)
* [Running Experiments](#running-experiments)
* [Pretraining Corpus](#pretraining-corpus)
* [IndicGLUE](#iglue)
* [Additional Evaluation Tasks](#additional-evaluation-tasks)
* [Evaluation Results](#evaluation-results)
* [Downloads](#downloads)
* [Citing](#citing)
* [License](#license)
* [Contributors](#contributors)
* [Contact](#contact)



### Introduction

The Indic BERT model is based on the ALBERT model, a recent derivative of BERT. It is pre-trained on 12 Indian languages: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu.

The easiest way to use Indic BERT is through the Huggingface transformers library. It can be simply loaded like this:

```python
# pip3 install transformers
# pip3 install sentencepiece

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert')
```
Note: To preserve accents (vowel matras / diacritics) while tokenization (Read this issue for more details [#26](../../issues/26) ), use this:
```python
tokenizer = transformers.AutoTokenizer.from_pretrained('ai4bharat/indic-bert', keep_accents=True)
```

### Setting up the Code

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
python3 -m fine_tune.cli --help
```

To evaluate a specific model with default hyper-parameters, execute:

```bash
python3 -m fine_tune.cli --model <model name> --dataset <dataset name> --lang <iso lang code> --iglue_dir <base path to indic glue dir> --output_dir <output dir>
```

For more advanced usage of the fine-tuning code, refer [this document](https://github.com/AI4Bharat/indic-bert/blob/master/docs/advanced-usage.md).

### Pretraining Corpus

We pre-trained indic-bert on AI4Bharat's monolingual corpus. The corpus has the following distribution of languages:


| Language          | as     | bn     | en     | gu     | hi     | kn     |         |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| **No. of Tokens** | 36.9M  | 815M   | 1.34B  | 724M   | 1.84B  | 712M   |         |
| **Language**      | **ml** | **mr** | **or** | **pa** | **ta** | **te** | **all** |
| **No. of Tokens** | 767M   | 560M   | 104M   | 814M   | 549M   | 671M   | 8.9B    |



### IndicGLUE

IGLUE is a natural language understanding benchmark for Indian languages that we propose. While building this benchmark, our objective was also to cover most of the 11 Indian languages for each task. It consists of the following tasks:

##### News Category Classification

Predict the genre of a given news article. The dataset contains around 125k news articles across 9 Indian languages. Example:

*Article Snippet*:

```
கர்நாடக சட்டப் பேரவையில் வெற்றி பெற்ற எம்எல்ஏக்கள் இன்று பதவியேற்றுக் கொண்ட நிலையில் , காங்கிரஸ் எம்எல்ஏ ஆனந்த் சிங் க்கள் ஆப்சென்ட் ஆகி அதிர்ச்சியை ஏற்படுத்தியுள்ளார் . உச்சநீதிமன்ற உத்தரவுப்படி இன்று மாலை முதலமைச்சர் எடியூரப்பா இன்று நம்பிக்கை வாக்கெடுப்பு நடத்தி பெரும்பான்மையை நிரூபிக்க உச்சநீதிமன்றம் உத்தரவிட்டது . 
```

*Category*: Politics



##### Named Entity Recognition

Recognize entities and their coarse types in a sequence of words. The dataset contains around 787k examples across 11 Indian languages.

*Example*:


|     |     |     |     |     |     |     |     |     |     |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Token** | चाणक्य | पुरी   | को   | यहाँ  | देखने  | हेतु   | यहाँ  | क्लिक | करें   |
| **Type** | B-LOC | I-LOC | O    | O    | O    | O    | O    | O    | O    |



##### Headline Prediction

 Predict the correct headline for a news article from a given list of four candidate headlines. The dataset contains around 880k examples across 11 Indian languages. Example:

*News Article:*

     ರಾಷ್ಟ್ರೀಯ\nಪುಣೆ: 23 ವರ್ಷದ ಇನ್ಫೋಸಿಸ್ ಮಹಿಳಾ ಟೆಕ್ಕಿಯೊಬ್ಬರನ್ನು ನಡು ರಸ್ತೆಯಲ್ಲಿಯೇ ಮಾರಾಕಾಸ್ತ್ರಗಳಿಂದ ಬರ್ಬರವಾಗಿ ಹತ್ಯೆ ಮಾಡಿರುವ ಘಟನೆ ಪುಣೆಯಲ್ಲಿ ಶನಿವಾರ ರಾತ್ರಿ ನಡೆದಿದೆ.\nಅಂತರ ದಾಸ್ ಕೊಲೆಯಾದ ಮಹಿಳಾ ಟೆಕ್ಕಿಯಾಗಿದ್ದಾರೆ. ಅಂತರಾ ಅವರು ಪಶ್ಚಿಮ ಬಂಗಾಳದ ಮೂಲದವರಾಗಿದ್ದಾರೆ. ಕಳೆದ ರಾತ್ರಿ 8.00 ಗಂಟೆ ಸುಮಾರಿಗೆ ಕೆಲಸ ಮುಗಿಸಿ ಮನೆಗೆ ತೆರಳುತ್ತಿದ್ದ ಸಂದರ್ಭದಲ್ಲಿ ಅಂತರಾ ಅವರ ಮೇಲೆ ದಾಳಿ ಮಾಡಿರುವ ದುಷ್ಕರ್ಮಿಗಳು ಮಾರಾಕಾಸ್ತ್ರಗಳಿಂದ ಹಲ್ಲೆ ನಡೆಸಿದ್ದಾರೆಂದು ಪೊಲೀಸರು ಹೇಳಿದ್ದಾರೆ.\nದಾಳಿ ನಡೆಸಿದ ನಂತರ ರಕ್ತದ ಮಡುವಿನಲ್ಲಿ ಬಿದ್ದು ಒದ್ದಾಡುತ್ತಿದ್ದ ಅಂತರಾ ಅವರನ್ನು ಸ್ಥಳೀಯರು ಆಸ್ಪತ್ರೆಗೆ ದಾಳಸಿದ್ದಾರೆ. ಆದರೆ, ಆಸ್ಪತ್ರೆಗೆ ದಾಖಲಿಸುವಷ್ಟರಲ್ಲಿ ಅಂತರಾ ಅವರು ಸಾವನ್ನಪ್ಪಿದ್ದಾರೆಂದು ಅವರು ಹೇಳಿದ್ದಾರೆ.\nಪ್ರಕರಣ ದಾಖಲಿಸಿಕೊಂಡಿರುವ ಪೊಲೀಸರು ತನಿಖೆ ಆರಂಭಿಸಿದ್ದಾರೆ",
*Candidate 1*: ಇನ್ಫೋಸಿಸ್ ಮಹಿಳಾ ಟೆಕ್ಕಿಯ ಬರ್ಬರ ಹತ್ಯೆ   *[correct answer]*
*Candidate 2:* ಮಾನಸಿಕ ಅಸ್ವಸ್ಥೆ ಮೇಲೆ  ಮಕ್ಕಳ ಕಳ್ಳಿ ಎಂದು ಭೀಕರ ಹಲ್ಲೆ
*Candidate 3:* ಕಸಬ ಬೆಂಗ್ರೆಯಲ್ಲಿ ಮುಸುಕುಧಾರಿಗಳ ತಂಡದಿಂದ ಮೂವರು ಯುವಕರ ಮೇಲೆ ಹಲ್ಲೆ : ಓರ್ವ ಗಂಭೀರ
*Candidate 4:* ಕಣಿವೆ ರಾಜ್ಯದಲ್ಲಿ mobile ಬಂದ್, ಪ್ರಿಂಟಿಂಗ್ ಪ್ರೆಸ್ ಮೇಲೆ ದಾಳಿ



##### Wikipedia Section Title Prediction

Predict the correct title for a Wikipedia section from a given list of four candidate titles. The dataset has 400k examples across 11 Indian languages.

*Section Text*:

```
2005માં, જેકમેન નિર્માણ કંપની, સીડ પ્રોડકશન્સ ઊભી કરવા તેના લાંબાસમયના મદદનીશ જહોન પાલેર્મો સાથે જોડાયા, જેમનો પ્રથમ પ્રોજેકટ 2007માં વિવા લાફલિન હતો. જેકમેનની અભિનેત્રી પત્ની ડેબોરા-લી ફર્નેસ પણ કંપનીમાં જોડાઈ, અને પાલેર્મોએ પોતાના, ફર્નેસ અને જેકમેન માટે “ યુનિટી ” અર્થવાળા લખાણની આ ત્રણ વીંટીઓ બનાવી.[૨૭] ત્રણેયના સહયોગ અંગે જેકમેને જણાવ્યું કે “ મારી જિંદગીમાં જેમની સાથે મેં કામ કર્યું તે ભાગીદારો અંગે ડેબ અને જહોન પાલેર્મો અંગે હું ખૂબ નસીબદાર છું. ખરેખર તેથી કામ થયું. અમારી પાસે જુદું જુદું સાર્મથ્ય હતું. હું તે પસંદ કરતો હતો. I love it. તે ખૂબ ઉત્તેજક છે. ”[૨૮]ફોકસ આધારિત સીડ લેબલ, આમન્ડા સ્કિવેઈટઝર, કેથરિન ટેમ્બલિન, એલન મંડેલબમ અને જોય મરિનો તેમજ સાથે સિડની આધારિત નિર્માણ કચેરીનું સંચાલન કરનાર અલાના ફ્રીનો સમાવેશ થતાં કદમાં વિસ્તૃત બની. આ કંપીનોનો ઉદ્દેશ જેકમેનના વતનના દેશની સ્થાનિક પ્રતિભાને કામે લેવા મધ્યમ બજેટવાળી ફિલ્મો બનાવવાનો છે. 
```

*Candidate 1:* એકસ-મેન

*Candidate 2:* કારકીર્દિ

*Candidate 3:* નિર્માણ કંપન [*correct answer*]

*Candidate 4:* ઓસ્ટ્રેલિય



##### Cloze-style Question Answering (WCQA)

Given a text with an entity randomly masked, the task is to predict that masked entity from a list of 4 candidate entities. The dataset contains around 239k examples across 11 languages. Example:

*Text*

```markdown
ਹੋਮੀ ਭਾਬਾ ਦਾ ਜਨਮ 1949 ਈ ਨੂਂ ਮੁੰਬਈ ਵਿੱਚ ਪਾਰਸੀ ਪਰਿਵਾਰ ਵਿੱਚ ਹੋਇਆ । ਸੇਂਟ ਮੇਰੀ ਤੋਂ ਮੁਢਲੀ ਸਿਖਿਆ ਪ੍ਰਾਪਤ ਕਰਕੇ ਉਹ ਬੰਬੇ ਯੂਨੀਵਰਸਿਟੀ ਗ੍ਰੈਜੁਏਸ਼ਨ ਲਈ ਚਲਾ ਗਿਆ । ਇਸ ਤੋਂ ਬਾਅਦ ਉਹ ਉਚੇਰੀ ਸਿਖਿਆ ਲਈ <MASK> ਚਲਾ ਗਿਆ । ਉਸਨੇ ਓਥੇ ਆਕਸਫੋਰਡ ਯੂਨੀਵਰਸਿਟੀ ਤੋਂ ਐਮ.ਏ ਅਤੇ ਐਮ ਫਿਲ ਦੀਆਂ ਡਿਗਰੀਆਂ ਪ੍ਰਾਪਤ ਕੀਤੀਆਂ । ਤਕਰੀਬਨ ਦਸ ਸਾਲ ਤਕ ਉਸਨੇ ਸੁਸੈਕਸ ਯੂਨੀਵਰਸਿਟੀ ਦੇ ਅੰਗਰੇਜ਼ੀ ਵਿਭਾਗ ਵਿੱਚ ਬਤੌਰ ਲੈਕਚਰਾਰ ਕਾਰਜ ਨਿਭਾਇਆ । ਇਸਤੋਂ ਇਲਾਵਾ ਹੋਮੀ ਭਾਬਾ ਪੈਨਸੁਲਵੇਨਿਆ , ਸ਼ਿਕਾਗੋ ਅਤੇ ਅਮਰੀਕਾ ਦੀ ਹਾਰਵਰਡ ਯੂਨੀਵਰਸਿਟੀ ਵਿੱਚ ਵੀ ਪ੍ਰੋਫ਼ੇਸਰ ਦੇ ਆਹੁਦੇ ਤੇ ਰਿਹਾ ।
```

*Candidate 1*: ਬਰਤਾਨੀਆ *[correct answer]*
*Candidate 2*: ਭਾਰਤ
*Candidate 3*: ਸ਼ਿਕਾਗੋ
*Candidate 4*: ਪਾਕਿਸਤਾਨ



##### Cross-lingual Sentence Retrieval (XSR)

Given a sentence in language $L_1$ the task is to retrieve its translation from a set of candidate sentences in language $L_2$. The dataset contains around 39k parallel sentence pairs across 8 Indian languages. Example:

*Input Sentence*

```
In the health sector the nation has now moved ahead from the conventional approach.
```

*Retrieve the following translation from a set of 4886 sentences:*

```
ആരോഗ്യമേഖലയില് ഇന്ന് രാജ്യം പരമ്പരാഗത രീതികളില് നിന്ന് മുന്നേറിക്കഴിഞ്ഞു.
```



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



### Evaluation Results


##### IndicGLUE

Task | mBERT | XLM-R | IndicBERT
-----| ----- | ----- | ------ 
News Article Headline Prediction | 89.58 | 95.52 | **95.87** 
Wikipedia Section Title Prediction| **73.66** | 66.33 | 73.31 
Cloze-style multiple-choice QA | 39.16 | 27.98 | **41.87** 
Article Genre Classification | 90.63 | 97.03 | **97.34** 
Named Entity Recognition (F1-score) | **73.24** | 65.93 | 64.47 
Cross-Lingual Sentence Retrieval Task | 21.46 | 13.74 | **27.12** 
Average | 64.62 | 61.09 | **66.66** 

##### Additional Tasks


Task | Task Type | mBERT | XLM-R | IndicBERT 
-----| ----- | ----- | ------ | ----- 
BBC News Classification | Genre Classification | 60.55 | **75.52** | 74.60 
IIT Product Reviews | Sentiment Analysis | 74.57 | **78.97** | 71.32 
IITP Movie Reviews | Sentiment Analaysis | 56.77 | **61.61** | 59.03 
Soham News Article | Genre Classification | 80.23 | **87.6** | 78.45 
Midas Discourse | Discourse Analysis | 71.20 | **79.94** | 78.44 
iNLTK Headlines Classification | Genre Classification | 87.95 | 93.38 | **94.52** 
ACTSA Sentiment Analysis | Sentiment Analysis | 48.53 | 59.33 | **61.18** 
Winograd NLI | Natural Language Inference | 56.34 | 55.87 | **56.34** 
Choice of Plausible Alternative (COPA) | Natural Language Inference | 54.92 | 51.13 | **58.33** 
Amrita Exact Paraphrase | Paraphrase Detection | **93.81** | 93.02 | 93.75 
Amrita Rough Paraphrase | Paraphrase Detection | 83.38 | 82.20 | **84.33** 
Average |  |  69.84 | **74.42** | 73.66 


\* Note: all models have been restricted to a max_seq_length of 128.



### Downloads

The model can be downloaded [here](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/models/indic-bert-v1.tar.gz). Both tf checkpoints and pytorch binaries are included in the archive. Alternatively, you can also download it from [Huggingface](https://huggingface.co/ai4bharat/indic-bert).



### Citing

If you are using any of the resources, please cite the following article:

```
@inproceedings{kakwani2020indicnlpsuite,
    title={{IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages}},
    author={Divyanshu Kakwani and Anoop Kunchukuttan and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    booktitle={Findings of EMNLP},
}
```

We would like to hear from you if:

- You are using our resources. Please let us know how you are putting these resources to use.
- You have any feedback on these resources.



### License

The IndicBERT code (and models) are released under the MIT License.

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
- Pratyush Kumar ([pratyush@cse.iitm.ac.in](mailto:pratyush@cse.iitm.ac.in))

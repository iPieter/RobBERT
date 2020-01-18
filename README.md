# RobBERT

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

A Dutch language model based on [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) with some tasks specific to Dutch.

<img src="./res/robbert_logo.png" alt="RobBERT logo" width="300"/>

## Getting started

RobBERT can easily be used in two different ways, namely either using [Fairseq RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) code or using [HuggingFace Transformers](https://github.com/huggingface/transformers)

### Using Huggingface transformers

(to do: write)


### Using Fairseq

You can also use RobBERT using the RoBERTa architecture code.
We use the same encoder and dictionary as [Fairseq](https://github.com/pytorch/fairseq/), which you can download like this (use curl on Mac OS): 

```
mkdir data
cd data
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

Then download our model from [todo]

You can then use all functions of [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta), but using RobBERT's model.pt.

## Heads

We also provide the following fine-tuned heads:

- Prediction of `die` or `dat` in sentences as a classification problem (trained on [EuroParl](http://www.statmt.org/europarl/) sentences)
- Sentiment analysis (trained on [DBRD](https://github.com/benjaminvdb/110kDBRD))

## Experiments from paper

You can replicate the experiments done in our paper by following the following steps.
You can install the required dependencies either the requirements.txt or pipenv:
- Installing the dependencies from the requirements.txt file using `pip install -r requirements.txt`
- OR install using [Pipenv](https://pipenv.readthedocs.io/en/latest/) *(install by running `pip install pipenv` in your terminal)* by running `pipenv install`.


### Classification
In this section we describe how to use the scripts we provide to fine-tune models, which should be general enough to reuse for other desired textual classification tasks.

#### Sentiment analysis using the Dutch Book Review Dataset

- Download the Dutch book review dataset from [https://github.com/benjaminvdb/110kDBRD](https://github.com/benjaminvdb/110kDBRD), and save it to `data/raw/110kDBRD`
- Run `src/preprocess_dbrd.py`
- (TODO: write)

#### Predicting the Dutch pronouns _die_ and _dat_
We fine-tune our model on the Dutch [Europarl corpus](http://www.statmt.org/europarl/). You can download it first with:

```
cd data\raw\europarl\
wget -N 'http://www.statmt.org/europarl/v7/nl-en.tgz'
tar zxvf nl-en.tgz
```
As a sanity check, now you should have the following files in your `data/raw/europarl` folder:

```
europarl-v7.nl-en.en
europarl-v7.nl-en.nl
nl-en.tgz
```

Then you can run the preprocessing with the following script, which fill first process the Europarl corpus to remove sentences without any _die_ or _dat_.
Afterwards, it will flip the pronoun and join both sentences together with a `<sep>` token.

```
python src/preprocess_diedat.py data/europarl-v7.nl-en.nl
./preprocess_diedat.sh
```

note: You can monitor the progress of the first preprocessing step with `watch -n 2 wc -l data/europarl-v7.nl-en.nl.sentences`. This will take a while, but it's certainly not needed to use all inputs. This is after all why you want to use a pre-trained language model. You can terminate the python script at any time and the second step will only use those._

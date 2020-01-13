# RobBERT

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

A Dutch language model based on RoBERTa with some tasks specific to Dutch. For now, we also provide the following fine-tuned heads:

- Prediction of `die` or `dat` in sentences. Trained on 10k sentences.


# Getting started
To get started, make sure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed *(if not, just run `pip install pipenv` in your terminal)*.
Afterwards, you can install the dependencies using the following command:

```
pipenv install
```

We use byte pair encodings (BPE), i.e. the same encoder and dictionary as [Fairseq](https://github.com/pytorch/fairseq/), which you can download like this (use curl on mac os): 

```
mkdir data
cd data
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

# Downstream tasks: Fine-tuning the model
In this section we describe how to use the scripts we provide to fine-tune models, hopefully this will be general enough to reuse for other tasks.

## Classification

### Predicting the Dutch pronouns _die_ and _dat_
We fine-tune our model on the Dutch [Europarl corpus](http://www.statmt.org/europarl/). You can download it first with:

```
cd data\
wget -N 'http://www.statmt.org/europarl/v7/nl-en.tgz'
tar zxvf nl-en.tgz
```
As a sanity check, now you should have the following files in your `data\` folder:

```
dict.txt
encoder.json
europarl-v7.nl-en.en
europarl-v7.nl-en.nl
nl-en.tgz
vocab.bpe
```

Then you can run the preprocessing with the following script, which fill first process the Europarl corpus to remove sentences without any _die_ or _dat_. Afterwards, it will flip the pronoun and join both sentences together with a `<sep>` token.

```
python src/preprocess_diedat.py data/europarl-v7.nl-en.nl
./preprocess_diedat.sh
```

_note: You can monitor the progress of the first preprocessing step with `watch -n 2 wc -l data/europarl-v7.nl-en.nl.sentences`. This will take a while, but it's certainly not needed to use all inputs. This is after all why you want to use a pre-trained language model. You can terminate the python script at any time and the second step will only use those._

## Tagging

### POS tagging
- Download the [Universal Dependencies LASSY small dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3105).
- Unzip the file, then unzip `ud-treebanks-v2.5`, then place the `UD_Dutch-LassySmall` such that
`/data/raw/UD_Dutch-LassySmall` is the main folder
- Run `preprocess_lassy_ud.py`

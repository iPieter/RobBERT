---
language: "nl"
thumbnail: "https://github.com/iPieter/RobBERT/raw/master/res/robbert_logo.png"
tags:
- Dutch
- Flemish
- RoBERTa
- RobBERT
license: mit
datasets:
- oscar
- oscar (NL)
- dbrd
- lassy-ud
- europarl-mono
- conll2002
widget:
- text: "Hallo, ik ben RobBERT, een <mask> taalmodel van de KU Leuven."
---

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/raw/master/res/robbert_logo_with_name.png" alt="RobBERT: A Dutch RoBERTa-based Language Model" width="75%">
 </p>

# RobBERT: Dutch RoBERTa-based Language Model.

[RobBERT](https://github.com/iPieter/RobBERT) is the state-of-the-art Dutch BERT model. It is a large pre-trained general Dutch language model that can be fine-tuned on a given dataset to perform any text classification, regression or token-tagging task. As such, it has been successfully used by many [researchers](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=7180110604335112086) and [practitioners](https://huggingface.co/models?search=robbert) for achieving state-of-the-art performance for a wide range of Dutch natural language processing tasks, including:

- [Emotion detection](https://www.aclweb.org/anthology/2021.wassa-1.27/)
- Sentiment analysis ([book reviews](https://arxiv.org/pdf/2001.06286.pdf), [news articles](https://biblio.ugent.be/publication/8704637/file/8704638.pdf)*)
- [Coreference resolution](https://arxiv.org/pdf/2001.06286.pdf)
- Named entity recognition ([CoNLL](https://arxiv.org/pdf/2001.06286.pdf), [job titles](https://arxiv.org/pdf/2004.02814.pdf)*, [SoNaR](https://github.com/proycon/deepfrog))
- Part-of-speech tagging ([Small UD Lassy](https://arxiv.org/pdf/2001.06286.pdf), [CGN](https://github.com/proycon/deepfrog))
- [Zero-shot word prediction](https://arxiv.org/pdf/2001.06286.pdf)
- [Humor detection](https://arxiv.org/pdf/2010.13652.pdf)
- [Cyberbulling detection](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/automatic-classification-of-participant-roles-in-cyberbullying-can-we-detect-victims-bullies-and-bystanders-in-social-media-text/A2079C2C738C29428E666810B8903342)
- [Correcting dt-spelling mistakes](https://gitlab.com/spelfouten/dutch-simpletransformers/)*

and also achieved outstanding, near-sota results for:

- [Natural language inference](https://arxiv.org/pdf/2101.05716.pdf)*
- [Review classification](https://medium.com/broadhorizon-cmotions/nlp-with-r-part-5-state-of-the-art-in-nlp-transformers-bert-3449e3cd7494)*

\* *Note that several evaluations use RobBERT-v1, and that the second and improved RobBERT-v2 outperforms this first model on everything we tested*

*(Also note that this list is not exhaustive. If you used RobBERT for your application, we are happy to know about it! Send us a mail, or add it yourself to this list by sending a pull request with the edit!)*

More in-depth information about RobBERT can be found in our [blog post](https://people.cs.kuleuven.be/~pieter.delobelle/robbert/), [our paper](https://arxiv.org/abs/2001.06286) and [the RobBERT Github repository](https://github.com/iPieter/RobBERT)


## How to use

RobBERT uses the [RoBERTa](https://arxiv.org/abs/1907.11692) architecture and pre-training but with a Dutch tokenizer and training data. RoBERTa is the robustly optimized English BERT model, making it even more powerful than the original BERT model. Given this same architecture, RobBERT can easily be finetuned and inferenced using [code to finetune RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) models and most code used for BERT models, e.g. as provided by [HuggingFace Transformers](https://huggingface.co/transformers/) library.

By default, RobBERT has the masked language model head used in training. This can be used as a zero-shot way to fill masks in sentences. It can be tested out for free on [RobBERT's Hosted infererence API of Huggingface](https://huggingface.co/pdelobelle/robbert-v2-dutch-base?text=De+hoofdstad+van+Belgi%C3%AB+is+%3Cmask%3E.). You can also create a new prediction head for your own task by using any of HuggingFace's [RoBERTa-runners](https://huggingface.co/transformers/v2.7.0/examples.html#language-model-training), [their fine-tuning notebooks](https://huggingface.co/transformers/v4.1.1/notebooks.html) by changing the model name to `pdelobelle/robbert-v2-dutch-base`, or use the original fairseq [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) training regimes.

Use the following code to download the base model and finetune it yourself, or use one of our finetuned models (documented on  [our project site](https://people.cs.kuleuven.be/~pieter.delobelle/robbert/)).

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robbert-v2-dutch-base")
```

Starting with `transformers v2.4.0` (or installing from source), you can use AutoTokenizer and AutoModel.
You can then use most of [HuggingFace's BERT-based notebooks](https://huggingface.co/transformers/v4.1.1/notebooks.html) for finetuning RobBERT on your type of Dutch language dataset.


## Technical Details From The Paper


### Our Performance Evaluation Results

All experiments are described in more detail in our [paper](https://arxiv.org/abs/2001.06286), with the code in [our GitHub repository](https://github.com/iPieter/RobBERT).

### Sentiment analysis
Predicting whether a review is positive or negative using the [Dutch Book Reviews Dataset](https://github.com/benjaminvdb/110kDBRD).

|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| ULMFiT            | 93.8                     |
| BERTje            | 93.0                     |
| RobBERT v2        | **95.1**                 |

### Die/Dat (coreference resolution)

We measured how well the models are able to do coreference resolution by predicting whether "die" or "dat" should be filled into a sentence.
For this, we used the [EuroParl corpus](https://www.statmt.org/europarl/).

#### Finetuning on whole dataset

|   Model           | Accuracy [%]             |  F1 [%]    |
|-------------------|--------------------------|--------------|
| [Baseline](https://arxiv.org/abs/2001.02943) (LSTM)   |                          | 75.03        |
| mBERT             | 98.285                   | 98.033       |
| BERTje            | 98.268                   | 98.014       |
| RobBERT v2        | **99.232**               | **99.121**   |

#### Finetuning on 10K examples

We also measured the performance using only 10K training examples.
This experiment clearly illustrates that RobBERT outperforms other models when there is little data available.

|   Model           | Accuracy [%]             |  F1 [%]      |
|-------------------|--------------------------|--------------|
| mBERT             | 92.157                   | 90.898       |
| BERTje            | 93.096                   | 91.279       |
| RobBERT v2        | **97.816**               | **97.514**   |

#### Using zero-shot word masking task

Since BERT models are pre-trained using the word masking task, we can use this to predict whether "die" or "dat" is more likely.
This experiment shows that RobBERT has internalised more information about Dutch than other models.

|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| ZeroR             | 66.70                    |
| mBERT             | 90.21                    |
| BERTje            | 94.94                    |
| RobBERT v2        | **98.75**                |

### Part-of-Speech Tagging.

Using the [Lassy UD dataset](https://universaldependencies.org/treebanks/nl_lassysmall/index.html).


|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| Frog              | 91.7                     |
| mBERT             | **96.5**                 |
| BERTje            | 96.3                     |
| RobBERT v2        | 96.4                     |

Interestingly, we found that when dealing with **small data sets**, RobBERT v2 **significantly outperforms** other models.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/raw/master/res/robbert_pos_accuracy.png" alt="RobBERT's performance on smaller datasets">
 </p>

### Named Entity Recognition

Using the [CoNLL 2002 evaluation script](https://www.clips.uantwerpen.be/conll2002/ner/).


|   Model           | Accuracy [%]             |
|-------------------|--------------------------|
| Frog              | 57.31                    |
| mBERT             | **90.94**                |
| BERT-NL           | 89.7                     |
| BERTje            | 88.3                     |
| RobBERT v2        | 89.08                    |


## Pre-Training Procedure Details

We pre-trained RobBERT using the RoBERTa training regime.
We pre-trained our model on the Dutch section of the [OSCAR corpus](https://oscar-corpus.com/), a large multilingual corpus which was obtained by language classification in the Common Crawl corpus.
This Dutch corpus is 39GB large, with 6.6 billion words spread over 126 million lines of text, where each line could contain multiple sentences, thus using more data than concurrently developed Dutch BERT models.


RobBERT shares its architecture with [RoBERTa's base model](https://github.com/pytorch/fairseq/tree/master/examples/roberta), which itself is a replication and improvement over BERT.
Like BERT, it's architecture consists of 12 self-attention layers with 12 heads with 117M trainable parameters.
One difference with the original BERT model is due to the different pre-training task specified by RoBERTa, using only the MLM task and not the NSP task.
During pre-training, it thus only predicts which words are masked in certain positions of given sentences.
The training process uses the Adam optimizer with polynomial decay of the learning rate l_r=10^-6 and a ramp-up period of 1000 iterations, with hyperparameters beta_1=0.9
and RoBERTa's default beta_2=0.98.
Additionally, a weight decay of 0.1 and a small dropout of 0.1 helps prevent the model from overfitting. 


RobBERT was trained on a computing cluster with 4 Nvidia P100 GPUs per node, where the number of nodes was dynamically adjusted while keeping a fixed batch size of 8192 sentences.
At most 20 nodes were used (i.e. 80 GPUs), and the median was 5 nodes.
By using gradient accumulation, the batch size could be set independently of the number of GPUs available, in order to maximally utilize the cluster.
Using the [Fairseq library](https://github.com/pytorch/fairseq/tree/master/examples/roberta), the model trained for two epochs, which equals over 16k batches in total, which took about three days on the computing cluster.
In between training jobs on the computing cluster, 2 Nvidia 1080 Ti's also covered some parameter updates for RobBERT v2.


## Investigating Limitations and Bias

In the [RobBERT paper](https://arxiv.org/abs/2001.06286), we also investigated potential sources of bias in RobBERT.

We found that the zeroshot model estimates the probability of *hij* (he) to be higher than *zij* (she) for most occupations in bleached template sentences, regardless of their actual job gender ratio in reality.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/raw/master/res/gender_diff.png" alt="RobBERT's performance on smaller datasets">
 </p>

By augmenting the DBRB Dutch Book sentiment analysis dataset with the stated gender of the author of the review, we found that highly positive reviews written by women were generally more accurately detected by RobBERT as being positive than those written by men.

<p align="center"> 
    <img src="https://github.com/iPieter/RobBERT/raw/master/res/dbrd.png" alt="RobBERT's performance on smaller datasets">
 </p>



## How to Replicate Our Paper Experiments
Replicating our paper experiments is [described in detail on teh RobBERT repository README](https://github.com/iPieter/RobBERT#how-to-replicate-our-paper-experiments).

## Name Origin of RobBERT

Most BERT-like models have the word *BERT* in their name (e.g. [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html), [ALBERT](https://arxiv.org/abs/1909.11942), [CamemBERT](https://camembert-model.fr/), and [many, many others](https://huggingface.co/models?search=bert)).
As such, we queried our newly trained model using its masked language model to name itself *\<mask\>bert* using [all](https://huggingface.co/pdelobelle/robbert-v2-dutch-base?text=Mijn+naam+is+%3Cmask%3Ebert.) [kinds](https://huggingface.co/pdelobelle/robbert-v2-dutch-base?text=Hallo%2C+ik+ben+%3Cmask%3Ebert.) [of](https://huggingface.co/pdelobelle/robbert-v2-dutch-base?text=Leuk+je+te+ontmoeten%2C+ik+heet+%3Cmask%3Ebert.) [prompts](https://huggingface.co/pdelobelle/robbert-v2-dutch-base?text=Niemand+weet%2C+niemand+weet%2C+dat+ik+%3Cmask%3Ebert+heet.), and it consistently called itself RobBERT.
We thought it was really quite fitting, given that RobBERT is a [*very* Dutch name](https://en.wikipedia.org/wiki/Robbert) *(and thus clearly a Dutch language model)*, and additionally has a high similarity to its root architecture, namely [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html).

Since *"rob"* is a Dutch words to denote a seal, we decided to draw a seal and dress it up like [Bert from Sesame Street](https://muppet.fandom.com/wiki/Bert) for the [RobBERT logo](https://github.com/iPieter/RobBERT/blob/master/res/robbert_logo.png).

## Credits and citation

This project is created by [Pieter Delobelle](https://people.cs.kuleuven.be/~pieter.delobelle), [Thomas Winters](https://thomaswinters.be) and [Bettina Berendt](https://people.cs.kuleuven.be/~bettina.berendt/).
If you would like to cite our paper or model, you can use the following BibTeX:

```
@inproceedings{delobelle2020robbert,
    title = "{R}ob{BERT}: a {D}utch {R}o{BERT}a-based {L}anguage {M}odel",
    author = "Delobelle, Pieter  and
      Winters, Thomas  and
      Berendt, Bettina",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.292",
    doi = "10.18653/v1/2020.findings-emnlp.292",
    pages = "3255--3265"
}
```

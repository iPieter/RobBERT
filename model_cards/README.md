---
language: "nl"
thumbnail: "https://github.com/iPieter/RobBERT/blob/master/res/robbert_logo.png"
tags:
- Dutch
- RoBERTa
- RobBERT
license: "MIT
datasets:
- OSCAR
metrics:
- array of metric identifiers
---

# RobBERT

## Model description

[RobBERT](https://github.com/iPieter/RobBERT) is a Dutch state-of-the-art [RoBERTa](https://arxiv.org/abs/1907.11692)-based language model.

## Intended uses & limitations

#### How to use

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robBERT-base")
```

#### Limitations and bias

Provide examples of latent issues and potential remediations.

## Training data

Describe the data you used to train the model.
If you initialized it with pre-trained weights, add a link to the pre-trained model card or repository with description of the pre-training data.

## Training procedure

Preprocessing, hardware used, hyperparameters...

## Eval results



### BibTeX entry and citation info

```bibtex
@misc{delobelle2020robbert,
    title={RobBERT: a Dutch RoBERTa-based Language Model},
    author={Pieter Delobelle and Thomas Winters and Bettina Berendt},
    year={2020},
    eprint={2001.06286},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
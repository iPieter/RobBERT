#!/bin/bash

rm data/labels/dict.txt

for SPLIT in train dev; do
    python src/multiprocessing_bpe_encoder.py\
        --encoder-json data/encoder.json \
        --vocab-bpe data/vocab.bpe \
        --inputs "data/processed/diedat/$SPLIT.sentences" \
        --outputs "data/processed/diedat/$SPLIT.sentences.bpe" \
        --workers 24 \
        --keep-empty
done

fairseq-preprocess \
    --only-source \
    --trainpref "data/processed/diedat/train.sentences.bpe" \
    --validpref "data/processed/diedat/dev.sentences.bpe" \
    --destdir "data/input0" \
    --workers 24 \
    --srcdict data/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "data/processed/diedat/train.labels" \
    --validpref "data/processed/diedat/dev.labels" \
    --destdir "data/diedat/labels" \
    --workers 24


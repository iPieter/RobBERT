#!/bin/bash

split -l $[ $(wc -l data/europarl-v7.nl-en.nl.labels|cut -d" " -f1) * 70 / 100 ] data/europarl-v7.nl-en.nl.labels
mv xaa data/diedat/train.labels
mv xab data/diedat/dev.labels

split -l $[ $(wc -l data/europarl-v7.nl-en.nl.sentences|cut -d" " -f1) * 70 / 100 ] data/europarl-v7.nl-en.nl.sentences
mv xaa data/diedat/train.sentences
mv xab data/diedat/dev.sentences

rm data/labels/dict.txt

for SPLIT in train dev; do
    python src/multiprocessing_bpe_encoder.py\
        --encoder-json data/encoder.json \
        --vocab-bpe data/vocab.bpe \
        --inputs "data/diedat/$SPLIT.sentences" \
        --outputs "data/diedat/$SPLIT.sentences.bpe" \
        --workers 24 \
        --keep-empty
done

fairseq-preprocess \
    --only-source \
    --trainpref "data/diedat/train.sentences.bpe" \
    --validpref "data/diedat/dev.sentences.bpe" \
    --destdir "data/input0" \
    --workers 24 \
    --srcdict data/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "data/diedat/train.labels" \
    --validpref "data/diedat/dev.labels" \
    --destdir "data/diedat/labels" \
    --workers 24


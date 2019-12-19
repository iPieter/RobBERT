#!/bin/bash

split -l $[ $(wc -l data/europarl-v7.nl-en.nl.labels|cut -d" " -f1) * 70 / 100 ] data/europarl-v7.nl-en.nl.labels
mv xaa data/train.labels
mv xab data/dev.labels

split -l $[ $(wc -l data/europarl-v7.nl-en.nl.sentences|cut -d" " -f1) * 70 / 100 ] data/europarl-v7.nl-en.nl.sentences
mv xaa data/train.sentences
mv xab data/dev.sentences

rm data/labels/dict.txt

for SPLIT in train dev; do
    python src/multiprocessing_bpe_encoder.py\
        --encoder-json data/encoder.json \
        --vocab-bpe data/vocab.bpe \
        --inputs "data/$SPLIT.sentences" \
        --outputs "data/$SPLIT.sentences.bpe" \
        --workers 24 \
        --keep-empty
done

fairseq-preprocess \
    --only-source \
    --trainpref "data/train.sentences.bpe" \
    --validpref "data/dev.sentences.bpe" \
    --destdir "data/input0" \
    --workers 24 \
    --srcdict data/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "data/train.labels" \
    --validpref "data/dev.labels" \
    --destdir "data/label" \
    --workers 24


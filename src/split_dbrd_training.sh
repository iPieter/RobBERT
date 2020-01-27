#!/bin/bash
mv train.sentences.txt traineval.sentences.txt
mv train.labels.txt traineval.labels.txt
echo "renamed files to traineval.*.txt"

head traineval.sentences.txt -n -500 > train.sentences.txt
head traineval.labtels.txt -n -500 > train.labels.txt
head traineval.sentences.txt -n 500 > eval.sentences.txt
head traineval.labels.txt -n 500 > eval.labels.txt

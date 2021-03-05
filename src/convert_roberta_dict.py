#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description=main.__doc__)

    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/UD_Dutch-LassySmall/")
    parser.add_argument(
        "--dict",
        help='path to dict.txt',
        required=True
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.json',
        required=True
    )

    parser.add_argument(
        "--output-vocab",
        type=str,
        help='Location for the output vocab.json',
        required=True
    )

    return parser

def load_roberta_mapping(file):
    "Returns a dict with the position of each word-id in the dict.txt file."

    # This file is basically an ordered count and we're not interested in the count.
    lines = {line.rstrip('\n').split()[0]: k for  k, line in enumerate(file)}
    return lines


def map_roberta(mapping, vocab):
    "Combine vocab.json and dict.txt contents."
    inverse_vocab = {str(v): k for k, v in vocab.items()}

    # We add 4 extra tokens, so they also need to be added to the position id
    EXTRA_TOKENS = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}
    offset = len(EXTRA_TOKENS)

    output_vocab = EXTRA_TOKENS
    for word_id, position in mapping.items():
        if word_id in inverse_vocab:
            output_vocab[inverse_vocab[word_id]] = position + offset
        else:
            print("not found: {}".format(word_id))
            output_vocab[word_id] = position + offset

    output_vocab['<mask>'] = len(output_vocab)

    for word in [ inverse_vocab[x] for x in (set([str(vocab[k]) for k in vocab])-set(mapping)-set(EXTRA_TOKENS.keys()))]:
        output_vocab[word] = len(output_vocab)

    return output_vocab

def main(args: argparse.Namespace):
    "Merge a vocab.json file with a dict.txt created by Fairseq."

    # First we load the dict file created by Fairseq's Roberta
    with open(args.dict, encoding="utf-8") as dict_fp:
        mapping = load_roberta_mapping(dict_fp)

    # Now we load the vocab file
    with open(args.vocab_bpe, encoding="utf-8") as vocab_fp:
        vocab = json.load(vocab_fp)

    output_vocab = map_roberta(mapping, vocab)

    with open(args.output_vocab, "w", encoding="utf-8") as output_fp:
        json.dump(output_vocab, output_fp, ensure_ascii=False)


if __name__ == '__main__':
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    main(args)
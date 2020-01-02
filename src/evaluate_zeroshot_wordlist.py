# Class for evaluating the base BERDT model on a word list classification task without any fine tuning.

import argparse
from pathlib import Path
from typing import List

from fairseq.models.roberta import RobertaModel

from src.wordlistfiller import WordListFiller


def evaluate(words: List[str], path: Path = None, model: RobertaModel = None):
    if not model:
        model = RobertaModel.from_pretrained(
            '../models/berdt',
            checkpoint_file='model.pt'
        )
        model.eval()

    wordlistfiller = WordListFiller(words, model=model)

    dataset_path = path if path is not None else models_path / ("-".join(words) + ".tsv")

    correct = 0
    total = 0

    with open(dataset_path) as input_file:
        for line in input_file:
            sentence, index = line.split('\t')
            expected = words[int(index.strip())]

            predicted = wordlistfiller.find_optimal_word(sentence)
            if predicted == expected:
                correct += 1
            total += 1

            print(str(100 * correct / total) + "% correct", correct, total, predicted, expected, sentence)

    return correct, total


def create_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the europarl corpus for the die-dat task."
    )
    parser.add_argument("--words", help="List of comma-separated words to disambiguate", type=str, default="die,dat")
    parser.add_argument("--path", help="Path to the evaluation data", metavar="path", default=None)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    models_path = Path("..", "data", "processed", "wordlist")

    evaluate([x.strip() for x in args.words.split(",")], args.path)

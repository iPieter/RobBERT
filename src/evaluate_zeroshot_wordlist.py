# Class for evaluating the base BERDT model on a word list classification task without any fine tuning.

import argparse
from pathlib import Path

from fairseq.models.roberta import RobertaModel

from src.wordlistfiller import WordListFiller


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

    berdt = RobertaModel.from_pretrained(
        '../models/berdt',
        checkpoint_file='model.pt'
    )
    berdt.eval()

    words = [x.strip() for x in args.words.split(",")]
    wordlistfiller = WordListFiller(words, model=berdt)

    output_path = args.path if args.path is not None else models_path / (args.words.replace(',', '-') + ".tsv")

    correct = 0
    incorrect = 0

    with open(output_path) as input_file:
        for line in input_file:
            sentence, index = line.split('\t')
            expected = words[int(index.strip())]

            predicted = wordlistfiller.find_optimal_word(sentence)
            if predicted == expected:
                correct += 1
            else:
                incorrect += 1

    print( str(100*correct/(correct+incorrect)) + "% correct" )

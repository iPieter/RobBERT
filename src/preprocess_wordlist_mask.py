import argparse
from pathlib import Path

from src.wordlistfiller import WordListFiller


def create_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the europarl corpus for the die-dat task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/europarl-v7.nl-en.nl")
    parser.add_argument("--words", help="List of comma-separated words to disambiguate", type=str, default="die,dat")
    parser.add_argument("--number", help="Number of examples in the output dataset", type=int, default=10000)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    models_path = Path("..", "data", "processed", "wordlist")
    models_path.mkdir(parents=True, exist_ok=True)

    words = [x.strip() for x in args.words.split(",")]
    wordlistfiller = WordListFiller(words)

    output_path = models_path / (args.words.replace(',', '-') + ".tsv")

    with open(output_path, mode='a') as output:
        with open(args.path) as input_file:
            number_of_lines_to_add = args.number
            for line in input_file:
                line = line.strip()
                sentences = wordlistfiller.occlude_target_words_index(line)
                number_of_sentences = len(sentences)

                for i in range(min(number_of_lines_to_add, number_of_sentences)):
                    sentence = sentences[i]
                    output.write(sentence[0] + "\t" + str(sentence[1]) + '\n')
                number_of_lines_to_add -= number_of_sentences

                if number_of_lines_to_add <= 0:
                    break

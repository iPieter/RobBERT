import random
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse


def replace_die_dat_full_sentence(line, flout, fout):
    count = 0
    for i, word in enumerate(nltk.word_tokenize(line)):
        tokens = nltk.word_tokenize(line)
        if word == "die":
            tokens[i] = "dat"
        elif word == "dat":
            tokens[i] = "die"
        elif word == "Dat":
            tokens[i] = "Die"
        elif word == "Die":
            tokens[i] = "dat"

        if word.lower() == "die" or word.lower() == "dat":
            choice = random.getrandbits(1)
            results = TreebankWordDetokenizer().detokenize(tokens)

            if choice:
                output = "{} <sep> {}".format(results, line)
            else:
                output = "{} <sep> {}".format(line, results)

            fout.write(output + "\n")
            flout.write(str(choice) + "\n")
            count += 1

    return count


def create_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the europarl corpus for the die-dat task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path", default="data/europarl-v7.nl-en.nl")
    parser.add_argument("--full_sentence",
                        help="Storing full sentences separated by a separator, or partial sentences on both sides of the word",
                        action="store_true")
    parser.add_argument("--number", help="Number of examples in the output dataset", type=int, default=10000)

    return parser


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    if args.full_sentence:
        with open(args.path + '.labels', mode='a') as labels_output:
            with open(args.path + '.sentences', mode='a') as sentences_output:
                with open(args.path) as fp:
                    lines_processed = 0
                    for line in fp:
                        line = line.replace('\n', '').replace('\r', '')
                        count = replace_die_dat_full_sentence(line, labels_output, sentences_output)
                        lines_processed += count
                        if lines_processed >= args.number:
                            break
    else:
        # TODO
        pass

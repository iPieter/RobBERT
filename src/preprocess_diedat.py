import random
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse


def replace_die_dat_full_sentence(line, flout, fout, token_format="{} <sep> {}"):
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
            tokens[i] = "Dat"
        if word.lower() == "die" or word.lower() == "dat":
            choice = random.getrandbits(1)
            results = TreebankWordDetokenizer().detokenize(tokens)

            if choice:
                output = token_format.format(results, line)
            else:
                output = token_format.format(line, results)

            fout.write(output + "\n")
            flout.write(str(choice) + "\n")
            count += 1

    return count


def create_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the europarl corpus for the die-dat task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="data/raw/europarl-v7.nl-en.nl")
    parser.add_argument("--number", help="Number of examples in the output dataset", type=int, default=10000)
    parser.add_argument("--format", help="The format of the separators, valid options are `roberta`, `transformers-bert` and `transformers-roberta`.", default="roberta")

    return parser


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # Since we have different token formats for bert, roberta and transformer's roberta, we use an argument.
    if args.format.lower() == "roberta":
        token_format = "{} <sep> {}"
    elif args.format.lower() == "transformers-bert":
        token_format = "[CLS] {} [SEP] {} [SEP]"
    elif args.format.lower() == "transformers-roberta":
        token_format = "<s> {} </s></s> {} </s>" # Yes that second </s> is correct ...
    else: 
        print("No valid token format, defaulting to roberta")
        token_format = "{} <sep> {}"

    with open(args.path + '.labels', mode='a') as labels_output:
        with open(args.path + '.sentences', mode='a') as sentences_output:
            with open(args.path) as fp:
                lines_processed = 0
                for line in fp:
                    line = line.replace('\n', '').replace('\r', '')
                    count = replace_die_dat_full_sentence(line, labels_output, sentences_output, token_format)
                    lines_processed += count
                    if lines_processed >= args.number:
                        break

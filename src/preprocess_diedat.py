import random
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse

def replace_die_dat(line, flout, fout):
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
            flout.write(str(choice) +"\n")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the europarl corpus for the die-dat task."
    )
    # TODO Specify your real parameters here.
    parser.add_argument("path", help="Path to the corpus file.", metavar="path")

    args = parser.parse_args()

    with open(args.path + '.labels', mode='a') as flout:

        with open(args.path + '.sentences', mode='a') as fout:

            with open(args.path) as fp:
                for line in fp: 
                    line = line.replace('\n', '').replace('\r', '')
                    replace_die_dat(line, flout, fout)

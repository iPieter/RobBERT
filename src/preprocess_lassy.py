import argparse
from pathlib import Path
from os import listdir
from os.path import isfile, join

tags = ["--", "adj", "bw", "let", "lid", "n", "spec", "tsw", "tw", "vg", "vnw", "vz", "ww"]


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the LASSY corpus for the POS-tagging task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/LassySmall/LP")
    parser.add_argument("--number", help="Number of words in the output dataset", type=int, default=10000000)

    return parser


def get_label_index(label_name):
    return tags.index(label_name)


def process_lassy(output, output_labels, path, max_words):
    words_processed = 0

    lp_files = [f for f in listdir(path) if isfile(join(path, f))]
    for file_name in lp_files:
        file_path = path / file_name
        with open(file_path) as file_to_read:
            lines_processed = 0
            for line in file_to_read:
                words_processed += 1

                word, main_word, stem, detailed_tag, simple_tag = line.split("\t")

                # TODO: Output words and tags correctly
                output.write(word.strip() + "\n")
                output_labels.write(str(get_label_index(simple_tag.strip())) + "\n")

                if words_processed >= max_words:
                    break
        if words_processed >= max_words:
            break
    pass


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    processed_data_path = Path("..", "data", "processed", "lassy")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    output_path = processed_data_path / "sentences.tsv"
    output_labels_path = processed_data_path / "labels.tsv"

    with open(output_path, mode='w') as output:
        with open(output_labels_path, mode='w') as output_labels:
            process_lassy(output, output_labels, Path(args.path), int(args.number))

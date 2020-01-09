import argparse
from pathlib import Path


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the LASSY corpus for the POS-tagging task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/LassySmall/LP")
    parser.add_argument("--number", help="Number of words in the output dataset", type=int, default=10000)

    return parser


def process_lassy(output_path, path):
    pass


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    processed_data_path = Path("..", "data", "processed", "lassy")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    output_path = processed_data_path / (args.words.replace(',', '-') + ".tsv")

    with open(output_path, mode='w') as output:
        process_lassy(output_path, args.path)
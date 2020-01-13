import argparse
from pathlib import Path

from src import preprocess_util

universal_pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
                      "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the LASSY corpus for the POS-tagging task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/UD_Dutch-LassySmall/")
    parser.add_argument(
        "--encoder-json",
        help='path to encoder.json',
        default="../models/robbert/encoder.json"
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
        default="../models/robbert/vocab.bpe"
    )

    return parser


def get_label_index(label_name):
    return universal_pos_tags.index(label_name)


def process_lassy_ud(arguments, type, processed_data_path, raw_data_path):
    output_sentences_path, output_labels_path, output_tokenized_sentences_path, output_tokenized_labels_path = preprocess_util.get_sequence_file_paths(
        processed_data_path, type)
    tokenizer = preprocess_util.get_tokenizer(arguments)

    with open(output_sentences_path, mode='w') as output_sentences:
        with open(output_labels_path, mode='w') as output_labels:
            with open(output_tokenized_sentences_path, mode='w') as output_tokenized_sentences:
                with open(output_tokenized_labels_path, mode='w') as output_tokenized_labels:

                    file_path = raw_data_path / ("nl_lassysmall-ud-" + type + ".conllu")
                    with open(file_path) as file_to_read:
                        # Add new line after seeing comments or new line
                        new_line = False
                        # For removing first blank line
                        has_content = False
                        for line in file_to_read:
                            if not line.startswith("#") and len(line.strip()) > 0:

                                if new_line and has_content:
                                    output_sentences.write("\n")
                                    output_labels.write("\n")
                                    output_tokenized_sentences.write("\n")
                                    output_tokenized_labels.write("\n")
                                    new_line = False
                                has_content = True

                                index, word, main_word, universal_pos, detailed_tag, details, number, english_tag, \
                                number_and_english_tag, space_after = line.split("\t")

                                # Write out normal word & label
                                word = word.strip()
                                label = str(get_label_index(universal_pos.strip()))
                                preprocess_util.write_sequence_word_label(word, label, tokenizer, output_sentences,
                                                                          output_labels,
                                                                          output_tokenized_sentences,
                                                                          output_tokenized_labels)

                            else:
                                new_line = True


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    processed_data_path = Path("..", "data", "processed", "lassy_ud")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    process_lassy_ud(args, 'train', processed_data_path, Path(args.path))
    process_lassy_ud(args, 'dev', processed_data_path, Path(args.path))
    process_lassy_ud(args, 'test', processed_data_path, Path(args.path))

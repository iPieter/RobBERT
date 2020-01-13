import argparse
from pathlib import Path

from src import preprocess_util

ner_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]

type_map = {
    "dev": "testa",
    "test": "testb",
    "train": "train",
}


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the CONLL2002 corpus for the NER task."
    )
    parser.add_argument("--path", help="Path to the corpus file.", metavar="path",
                        default="../data/raw/conll2002/")
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


def get_dataset_suffix(type):
    return type_map.get(type, None)


def get_label_index(label_name):
    return ner_tags.index(label_name)


def process_connl2002_ner(arguments, type, processed_data_path, raw_data_path):
    output_sentences_path, output_labels_path, output_tokenized_sentences_path, output_tokenized_labels_path = preprocess_util.get_sequence_file_paths(
        processed_data_path, type)
    tokenizer = preprocess_util.get_tokenizer(arguments)

    with open(output_sentences_path, mode='w') as output_sentences:
        with open(output_labels_path, mode='w') as output_labels:
            with open(output_tokenized_sentences_path, mode='w') as output_tokenized_sentences:
                with open(output_tokenized_labels_path, mode='w') as output_tokenized_labels:

                    dataset_suffix = get_dataset_suffix(type)
                    if dataset_suffix is None:
                        raise Exception("Invalid type", type)

                    file_path = raw_data_path / ("ned." + dataset_suffix)
                    with open(file_path) as file_to_read:
                        # Add new line after seeing comments or new line
                        has_content = False
                        for line in file_to_read:
                            if not line.startswith("-DOCSTART-") and len(line.strip()) > 0:
                                has_content = True

                                word, pos, ner = line.split(" ")

                                # Write out normal word & label
                                word = word.strip()
                                label = str(get_label_index(ner.strip()))
                                preprocess_util.write_sequence_word_label(word, label, tokenizer, output_sentences,
                                                                          output_labels,
                                                                          output_tokenized_sentences,
                                                                          output_tokenized_labels)
                            elif has_content:
                                output_sentences.write("\n")
                                output_labels.write("\n")
                                output_tokenized_sentences.write("\n")
                                output_tokenized_labels.write("\n")
                                has_content = False


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    processed_data_path = Path("..", "data", "processed", "conll2002")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    process_connl2002_ner(args, 'train', processed_data_path, Path(args.path))
    process_connl2002_ner(args, 'dev', processed_data_path, Path(args.path))
    process_connl2002_ner(args, 'test', processed_data_path, Path(args.path))

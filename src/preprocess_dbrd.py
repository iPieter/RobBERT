import argparse
from pathlib import Path
from os import listdir
from os.path import isfile, join


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess the Dutch Book Reviews Dataset corpus for the sentiment analysis tagging task."
    )
    parser.add_argument("--path", help="Path to the corpus folder.", metavar="path", default="data/raw/110kDBRD/")

    return parser


def get_file_id(a):
    return int(a.split('_')[0])


def get_files_of_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def add_content_and_label(file_location, output_sentences, output_labels, label):
    with open(file_location) as file:
        content = " ".join(file.readlines()).replace('\n', ' ').replace('\r', '')
        single_spaced_content = ' '.join(content.split())
        output_sentences.write(single_spaced_content + "\n")
        output_labels.write(str(label) + "\n")


def process_dbrd(raw_data_path, test_or_train):
    processed_data_path = Path("..", "data", "processed", "dbrd")
    processed_data_path.mkdir(parents=True, exist_ok=True)

    output_sentences_path = processed_data_path / (test_or_train + ".sentences.txt")
    output_labels_path = processed_data_path / (test_or_train + ".labels.txt")

    with open(output_sentences_path, mode='w') as output_sentences:
        with open(output_labels_path, mode='w') as output_labels:
            pos_files_folder = raw_data_path / test_or_train / 'pos'
            neg_files_folder = raw_data_path / test_or_train / 'neg'

            pos_files = get_files_of_folder(pos_files_folder)
            neg_files = get_files_of_folder(neg_files_folder)

            pos_files.sort(key=get_file_id)
            neg_files.sort(key=get_file_id)


            assert len(pos_files) == len(neg_files)

            # process file by intertwining the files, such that the model can learn better
            for i in range(len(pos_files)):
                add_content_and_label(pos_files_folder / pos_files[i], output_sentences, output_labels, 1)
                add_content_and_label(neg_files_folder / neg_files[i], output_sentences, output_labels, 0)


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    process_dbrd(Path(args.path), 'train')
    process_dbrd(Path(args.path), 'test')

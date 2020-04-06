from src.multiprocessing_bpe_encoder import MultiprocessingEncoder

seperator_token = "\t"


def get_sequence_file_paths(processed_data_path, type):
    output_sentences_path = processed_data_path / (type + ".sentences.tsv")
    output_labels_path = processed_data_path / (type + ".labels.tsv")
    output_tokenized_sentences_path = processed_data_path / (type + ".sentences.bpe")
    output_tokenized_labels_path = processed_data_path / (type + ".txt")
    return output_sentences_path, output_labels_path, output_tokenized_sentences_path, output_tokenized_labels_path


def write_sequence_word_label(word, label, tokenizer, output_sentences, output_labels, output_tokenized_sentences,
                              output_tokenized_labels):
    output_sentences.write(word.strip() + seperator_token)
    output_labels.write(label + seperator_token)

    # Write tokenized
    tokenized_word = tokenizer.encode(word.strip())
    output_tokenized_sentences.write(seperator_token.join(tokenized_word) + seperator_token)
    output_tokenized_labels.write(len(tokenized_word) * (label + seperator_token))


def get_tokenizer(arguments):
    tokenizer = MultiprocessingEncoder(arguments)
    tokenizer.initializer()
    return tokenizer
import os
import logging
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, model_name_or_path, file_path="train", block_size=512, overwrite_cache=True, mask_padding_with_zero=True):
        assert os.path.isfile(file_path + '.sentences')

        assert os.path.isfile(file_path + '.labels')

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, model_name_or_path + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path + ".labels", encoding="utf-8") as flabel:
                with open(file_path + ".sentences", encoding="utf-8") as f:
                    for sentence in f:
                        tokenized_text = tokenizer.encode(tokenizer.tokenize(sentence)[-block_size + 3 : -1])

                        input_mask = [1 if mask_padding_with_zero else 0] * len(tokenized_text)

                        pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                        while len(tokenized_text) < block_size:
                            tokenized_text.append(pad_token)
                            input_mask.append(0 if mask_padding_with_zero else 1)
                            #segment_ids.append(pad_token_segment_id)
                            #p_mask.append(1)

                        #self.examples.append([tokenizer.build_inputs_with_special_tokens(tokenized_text[0 : block_size]), [0], [0]])
                        label = next(flabel)
                        self.examples.append([tokenized_text[0 : block_size - 3], input_mask[0 : block_size - 3], [1] if label.startswith("1") else [0]])
                    
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return [torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]), torch.tensor([0]), torch.tensor(self.examples[item][2])]


def load_and_cache_examples(model_name_or_path, tokenizer, data_file):
    dataset = TextDataset(
        tokenizer,
        model_name_or_path,
        file_path=data_file,
        block_size=512
    )
    return dataset

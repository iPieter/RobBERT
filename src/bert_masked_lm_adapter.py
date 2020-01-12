# Class to mimick the fill_mask functionality of the RobertaModel, but for BERT models.
# Used to evaluate wordlistmasks
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer



class BertMaskedLMAdapter:
    def __init__(self, model_name=None, model=None, tokenizer=None):
        self._tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(model_name)
        self._model = model if model else BertForMaskedLM.from_pretrained(model_name)
        self._model.eval()

    def fill_mask(self, text, topk=4):
        text = text.replace("<mask>", "[MASK]")
        if not text.startswith("[CLS]"):
            text = "[CLS] " + text
        if not text.endswith("[SEP]"):
            text = text + " [SEP]"

        tokenized_text = self._tokenizer.tokenize(text)
        masked_index = tokenized_text.index('[MASK]')
        indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            predictions = self._model(tokens_tensor, segments_tensors)

        predicted_indices = list(np.argpartition(predictions[0][0][masked_index], -4)[-topk:])
        predicted_indices.reverse()
        return [self._convert_token(predicted_index) for predicted_index in predicted_indices]

    def _convert_token(self, predicted_index):
        text = self._tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return None, None, text if text.startswith("##") else (" " + text)


if __name__ == '__main__':
    from pathlib import Path

    from src import evaluate_zeroshot_wordlist
    mlm = BertMaskedLMAdapter(model_name='bert-base-multilingual-uncased')
    result = mlm.fill_mask("Er is een meisje <mask> daar loopt.", 1024)
    print(result)

    from src.wordlistfiller import WordListFiller

    wordfiller = WordListFiller(["die", "dat"], model=mlm)
    print(wordfiller.find_optimal_word("Er is een meisje <mask> daar loopt."))
    evaluate_zeroshot_wordlist.evaluate(["die", "dat"],
                                        path=Path("..", "data", "processed", "wordlist","die-dat.test.tsv"),
                                        odel=mlm, print_step=10)

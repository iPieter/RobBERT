from typing import List, Tuple

from fairseq.models.roberta import RobertaHubInterface
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class WordListFiller:
    def __init__(self, target_words: List[str],
                 model: RobertaHubInterface = None,
                 detokenizer: TreebankWordDetokenizer = TreebankWordDetokenizer(),
                 topk_limit=2048):
        self._model = model
        self._target_words = [x.lower().strip() for x in target_words]
        self._target_words_spaced = set([" " + word for word in self._target_words])
        self._detokenizer = detokenizer
        self._top_k_limit = topk_limit

    def find_optimal_word(self, text: str) -> str:
        if self._model is None:
            raise AttributeError("No model given to find the optimal word")
        topk = 4
        result = None
        while result is None and topk <= self._top_k_limit:
            filler_words = self._model.fill_mask(text, topk=topk)
            result = next((x[2].strip() for x in filler_words if x[2].lower() in self._target_words_spaced), None)
            topk *= 2
        return result

    # Transforms a sentence into a list of sentences with target words masked if the sentence contains this
    def occlude_target_words(self, input_sentence: str) -> List[Tuple[str, str]]:
        tokenized = word_tokenize(input_sentence)
        result = []
        for i in range(len(tokenized)):
            if tokenized[i] in self._target_words:
                new_sentence_tokens = tokenized[:i] + ["<mask>"] + tokenized[i + 1:]
                new_sentence = self._detokenizer.detokenize(new_sentence_tokens)
                result.append((new_sentence, tokenized[i]))
        return result

    def occlude_target_words_index(self, input_sentence: str) -> List[Tuple[str, int]]:
        return [(s[0], self._target_words.index(s[1].strip().lower())) for s in self.occlude_target_words(input_sentence)]
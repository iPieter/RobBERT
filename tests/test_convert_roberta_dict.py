import unittest
from io import StringIO

from src.convert_roberta_dict import load_roberta_mapping, map_roberta


class ConvertRobertaTestCase(unittest.TestCase):
    def test_load_roberta_mapping(self):
        # Create a mock file
        file = StringIO(initial_value="3 12\n1 8\n2 7")
        mapping = load_roberta_mapping(file)

        # test our little mapping
        self.assertEqual(mapping['3'], 0, msg="First element in dict.txt is 3, should have id = 0")
        self.assertEqual(mapping['1'], 1, msg="Second element in dict.txt is 1, should have id = 1")
        self.assertEqual(mapping['2'], 2, msg="Third element in dict.txt is 2, should have id = 2")

    def test_map_roberta(self):
        file = StringIO(initial_value="3 12\n1 8\n2 7")
        mapping = load_roberta_mapping(file)

        vocab = {"Een": 3, "Twee": 2, "Drie": 1}

        output_vocab = map_roberta(mapping, vocab)

        self.assertEqual(output_vocab['<s>'], 0, msg="Extra tokens")
        self.assertEqual(output_vocab['<pad>'], 1, msg="Extra tokens")
        self.assertEqual(output_vocab['</s>'], 2, msg="Extra tokens")
        self.assertEqual(output_vocab['<unk>'], 3, msg="Extra tokens")
        self.assertEqual(output_vocab['Een'], 4, msg="'Een' has vocab_id = 3, which is mapped to 0 (+4)")
        self.assertEqual(output_vocab['Twee'], 6, msg="'Twee' has vocab_id = 3, which is mapped to 2 (+4)")
        self.assertEqual(output_vocab['Drie'], 5, msg="'Drie' has vocab_id = 1, which is mapped to 1 (+4)")


    def test_map_roberta_unused_tokens(self):
        """
        The fast HuggingFace tokenizer requires that all tokens in the merges.txt are also present in the
        vocab.json. When converting a Fairseq dict.txt, this is not necessarily the case in a naive implementation.

        More info: https://github.com/huggingface/transformers/issues/9290
        """

        file = StringIO(initial_value="3 12\n1 8\n2 7")
        mapping = load_roberta_mapping(file)

        # Tokens "Vier" and "Vijf" are not used in the mapping (= dixt.txt)
        vocab = {"Een": 3, "Twee": 2, "Drie": 1, "Vier": 5, "Vijf": 4}

        output_vocab = map_roberta(mapping, vocab)

        self.assertEqual(output_vocab['<s>'], 0, msg="Extra tokens")
        self.assertEqual(output_vocab['<pad>'], 1, msg="Extra tokens")
        self.assertEqual(output_vocab['</s>'], 2, msg="Extra tokens")
        self.assertEqual(output_vocab['<unk>'], 3, msg="Extra tokens")
        self.assertEqual(output_vocab['Een'], 4, msg="'Een' has vocab_id = 3, which is mapped to 0 (+4)")
        self.assertEqual(output_vocab['Twee'], 6, msg="'Twee' has vocab_id = 3, which is mapped to 2 (+4)")
        self.assertEqual(output_vocab['Drie'], 5, msg="'Drie' has vocab_id = 1, which is mapped to 1 (+4)")
        self.assertIn(output_vocab['Vier'], [7, 8], msg="'Vier' has vocab_id = 5, which is mapped the next available value")
        self.assertIn(output_vocab['Vijf'], [8, 7], msg="'Vijf' has vocab_id = 4, which is mapped the next available value")

    def test_tokenization(self):
        sample_input = "De tweede poging: nog een test van de tokenizer met nummers."
        expected_output = [0, 62, 488, 5351, 30, 49, 9, 2142, 7, 5, 905, 859, 10605, 15, 3316, 4, 2]


if __name__ == '__main__':
    unittest.main()

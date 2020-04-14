import unittest
from io import StringIO

from src.convert_roberta_dict import load_roberta_mapping


class ConvertRobertaTestCase(unittest.TestCase):
    def test_load_roberta_mapping(self):
        # Create a mock file
        file = StringIO(initial_value="3 12\n1 8\n2 7")
        mapping = load_roberta_mapping(file)

        # test our little mapping
        self.assertEqual(mapping['3'], 0, msg="First element in dict.txt is 3, should have id = 0")
        self.assertEqual(mapping['1'], 1, msg="Second element in dict.txt is 1, should have id = 1")
        self.assertEqual(mapping['2'], 2, msg="Third element in dict.txt is 2, should have id = 2")

    def test_tokenization(self):
        sample_input = "De tweede poging: nog een test van de tokenizer met nummers."
        expected_output = [0, 62, 488, 5351, 30, 49, 9, 2142, 7, 5, 905, 859, 10605, 15, 3316, 4, 2]


if __name__ == '__main__':
    unittest.main()

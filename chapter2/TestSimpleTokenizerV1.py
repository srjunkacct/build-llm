import unittest

from chapter2.Preprocess import preprocess
from chapter2.SimpleTokenizerV1 import SimpleTokenizerV1


class MyTestCase(unittest.TestCase):
    def test_something(self):
        filePath = "/Users/edwardlee/git/build-llm/chapter2/the-verdict.txt"
        with open(filePath, "r", encoding="utf-8") as file:
            text = file.read()
        preprocessedWords = preprocess(text)
        sortedWords = sorted(set(preprocessedWords))
        vocabulary = {token:integer for integer,token in enumerate(sortedWords)}
        tokenizer = SimpleTokenizerV1(vocabulary)
        inputText=""""It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
        ids = tokenizer.encode(inputText)
        expectedIds = [1,56,2,850,988,602,533,746,5,1126,596,5,1,67,7,38,851,1108,754,793,7]
        self.assertEqual(expectedIds, ids)


if __name__ == '__main__':
    unittest.main()

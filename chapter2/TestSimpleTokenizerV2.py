import unittest


from Preprocess import preprocess
from SimpleTokenizerV2 import SimpleTokenizerV2


class MyTestCase(unittest.TestCase):
    def test_something(self):
        filePath = "/Users/edwardlee/git/build-llm/chapter2/the-verdict.txt"
        with open(filePath, "r", encoding="utf-8") as file:
            text = file.read()
        preprocessedWords = preprocess(text)
        sortedWords = sorted(list(set(preprocessedWords)))
        sortedWords.extend(["<|endoftext|>", "<|unk|>"])
        vocabulary = {token:integer for integer,token in enumerate(sortedWords)}
        tokenizer = SimpleTokenizerV2(vocabulary)
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        inputText = " <|endoftext|> ".join((text1, text2))
        ids = tokenizer.encode(inputText)
        expectedIds = [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
        self.assertEqual(expectedIds, ids)
        decodedText = "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
        self.assertEqual(decodedText, tokenizer.decode(ids))


if __name__ == '__main__':
    unittest.main()

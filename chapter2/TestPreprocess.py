import unittest

from chapter2.Preprocess import preprocess


class TestPreprocess(unittest.TestCase):
    def testPreprocess(self):
        with open("/Users/edwardlee/git/build-llm/chapter2/the-verdict.txt", "r", encoding="utf-8") as f:
            text = f.read()
        tokens = preprocess(text)
        self.assertEqual(4690, len(tokens))


if __name__ == '__main__':
    unittest.main()

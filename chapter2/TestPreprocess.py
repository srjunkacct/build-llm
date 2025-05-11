import os
import unittest

from Preprocess import preprocess
from pathlib import Path
import os


class TestPreprocess(unittest.TestCase):
    def testPreprocess(self):
        home = str(Path.home())
        with open(os.path.join(home,"git", "build-llm", "chapter2","the-verdict.txt"), "r", encoding="utf-8") as f:
            text = f.read()
        tokens = preprocess(text)
        self.assertEqual(4690, len(tokens))


if __name__ == '__main__':
    unittest.main()

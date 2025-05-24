import re

from chapter2.Preprocess import preprocess


class SimpleTokenizerV1:
    def __init__(self, vocabulary):
        self.string_to_integer = vocabulary
        self.integer_to_string = {i:s for s,i in vocabulary.items()}

    def encode(self, text):
        preprocessed = preprocess( text )
        ids = [self.string_to_integer[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.integer_to_string[i] for i in ids ])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


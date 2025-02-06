#!/usr/bin/python3

import re

def preprocess(input):

    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', input)
    tokens = [ item.strip() for item in tokens if item.strip() ]
    return tokens

if __name__ == "__main__":
    preprocess(input)


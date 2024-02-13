#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:33:35 2024

@author: martin
"""

import os
import sys
import json
import nltk
from time import time
from tqdm import tqdm
from datetime import timedelta
from multiprocessing import Pool
from PyMultiDictionary import MultiDictionary

LANGUAGES = ('French', 'Spanish', 'Portuguese', 'German',
             'Russian', 'Italian', 'Malaysian', 'Polish')
LANGUAGE_CODES = ('fr', 'es', 'pt', 'de', 'ru', 'it', 'ms', 'pl')


def translate(word):
    dictionary = MultiDictionary()
    translated = dictionary.translate('en', word)
    shortlist = [t for t in translated if t[0] in LANGUAGE_CODES]
    data = {word: shortlist}

    with open('results.jsonl', 'a', encoding='UTF-8') as f:
        f.write(json.dumps(data) + '\n')


def translate_words(n_words, n_processes):
    # extract words from corpus
    nltk.download('words')
    english = nltk.corpus.words.words()
    english = list(set([word.lower() for word in english]))

    # remove words that are already stored
    if os.path.exists('results.jsonl'):
        with open('results.jsonl', 'r') as f:
            json_list = list(f)
        data = [json.loads(json_str) for json_str in json_list]
        used_words = [list(d.keys())[0] for d in data]
        english = [e for e in english if e not in used_words]

        print(f'{len(used_words)} words already translated')
    else:
        with open('results.jsonl', 'a') as f:
            print('Making empty results file')

    print(f'{len(english)} words left to translate')

    start = time()
    n_items = len(english[:n_words])
    with Pool(processes=n_processes) as p:
        with tqdm(total=n_items) as pbar:
            for _ in p.imap_unordered(translate, english[:n_words]):
                pbar.update()

    end = time()
    strftime = timedelta(seconds=end - start)
    print(f'\nTime taken to translate {n_items} words:\t{strftime}')


if __name__ == '__main__':
    n_proc = min(os.cpu_count(), 8)
    if len(sys.argv) > 1:
        n_words = int(sys.argv[1])
    else:
        n_words = 1000
        print('No user-defined amount of words; Defaulting to 1000 words')
    data = translate_words(n_words, n_proc)

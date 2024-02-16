#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:33:35 2024

@author: martin
"""

import os
import sys
import nltk
import time
import pandas as pd
from tqdm import tqdm
from PyMultiDictionary import MultiDictionary

LANGUAGES = ('French', 'Spanish', 'Portuguese', 'German',
             'Russian', 'Italian', 'Malaysian', 'Polish')
LANGUAGE_CODES = ['fr', 'es', 'pt', 'de', 'ru', 'it', 'ms', 'pl']


def translate(word, delay):
    dictionary = MultiDictionary()
    translated = dictionary.translate('en', word)
    time.sleep(delay)
    shortlist = []
    for code in LANGUAGE_CODES:
        for t in translated:
            if t[0] == code:
                shortlist.append(t)
    if not shortlist:
        shortlist = [(0, '')] * 8
    data = pd.DataFrame(
        [[t[1] for t in shortlist]], columns=LANGUAGE_CODES, index=[0]
    )
    data['en'] = word
    data.to_csv('results.csv', encoding='UTF-8',
                index=False, mode='a', header=False)


def translate_words(n_words, delay):
    # extract words from corpus
    nltk.download('words')
    english = nltk.corpus.words.words()
    english = list(set([word.lower() for word in english]))

    # remove words that are already stored
    if os.path.exists('results.csv'):
        used_words = pd.read_csv(
            'results.csv', encoding='UTF-8', usecols=['en']
        )['en'].tolist()
        english = list(set(english) - set(used_words))

        print(f'{len(used_words)} words already translated')
    else:
        df = pd.DataFrame(list(), columns=LANGUAGE_CODES + ['en'])
        df.to_csv('results.csv', encoding='UTF-8', index=False)

    print(f'{len(english)} words left to translate')

    for word in tqdm(english[:n_words]):
        translate(word, delay)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n_words = int(sys.argv[1])
    else:
        n_words = 1000
        print('No user-defined amount of words; Defaulting to 1000 words')

    if len(sys.argv) > 2:
        delay = float(sys.argv[2])
        assert 1 > delay > 0, 'Delay should be a fraction of a second'
    else:
        delay = 0.1
        print('No user-defined delay time; Defaulting to 100ms')
    print('\n------ IF AN ERROR OCCURS, WAIT A FEW HOURS AND TRY AGAIN ------\n')
    translate_words(n_words, delay)

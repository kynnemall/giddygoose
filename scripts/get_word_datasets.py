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


def translate(word):
    dictionary = MultiDictionary()
    translated = dictionary.translate('en', word)
    time.sleep(0.5)
    shortlist = []
    for code in LANGUAGE_CODES:
        for t in translated:
            if t[0] == code:
                shortlist.append(t)
    if shortlist:
        data = pd.DataFrame(
            [[t[1] for t in shortlist]], columns=LANGUAGE_CODES, index=[0]
        )
        data['en'] = word
        data.to_csv('results.csv', encoding='UTF-8',
                    index=False, mode='a', header=False)


def translate_words(n_words):
    # extract words from corpus
    nltk.download('words')
    english = nltk.corpus.words.words()
    english = list(set([word.lower() for word in english]))

    # remove words that are already stored
    if os.path.exists('results.csv'):
        used_words = pd.read_csv(
            'results.csv', encoding='UTF-8', usecols=['en']
        )['en']
        english = [e for e in english if e not in used_words]

        print(f'{len(used_words)} words already translated')
    else:
        df = pd.DataFrame(list(), columns=LANGUAGE_CODES + ['en'])
        df.to_csv('results.csv', encoding='UTF-8', index=False)

    print(f'{len(english)} words left to translate')

    for word in tqdm(english[:n_words]):
        translate(word)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n_words = int(sys.argv[1])
    else:
        n_words = 1000
        print('No user-defined amount of words; Defaulting to 1000 words')
    print('\n------ IF AN ERROR OCCURS, WAIT A FEW HOURS AND TRY AGAIN ------\n')
    translate_words(n_words)

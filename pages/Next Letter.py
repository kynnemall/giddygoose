#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:04:14 2024

@author: martin
"""

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from tqdm import tqdm
from io import BytesIO
from supabase import create_client
from collections import Counter


@st.cache_data
def prepare_data():
    # download CSV from supabase
    url = st.secrets['SUPBASE_URL']
    key = st.secrets['SUPABASE_KEY']
    bucket_name = st.secrets['BUCKET']
    client = create_client(url, key)
    csv_bytes = client.storage.from_(bucket_name).download('translations.csv')
    df = pd.read_csv(BytesIO(csv_bytes))

    # process data
    df.dropna(inplace=True)
    for col in df.columns:
        df[col] = df[col].str.lower().str.replace(r'\W', '', regex=True)

    # count rows where at least N words are the same
    t = df.T
    keep_cols = [c for c in tqdm(t.columns) if t[c].nunique() > 2]
    df = t[keep_cols].T
    return df


def letter_frequencies(words):
    # count each character
    letters = []
    for word in words:
        letters.extend(list(word))
        letters.append(' ')  # add extra space for end of word

    letter_freq = Counter(letters)
    letters = pd.Series(letters)
    unique = letter_freq.keys()
    nchars = len(list(unique))
    mat = pd.DataFrame(
        np.zeros((nchars, nchars)), columns=unique, index=unique
    )

    for l in letter_freq.keys():
        if l != ' ':
            # get indexes of letters after the letter of interest
            idxs = letters[letters == l].index + 1

            # remove indexes larger than the possible values
            idxs = idxs[idxs < letters.size]
            next_ = letters[idxs]

            # calculate letter frequences and update `mat`
            freqs = next_.value_counts(normalize='freq')
            for next_l in freqs.index:
                mat.loc[l, next_l] = freqs[next_l]

    # remove space from index and columns
    index = [m for m in mat.index if m != ' ']
    mat.columns = [m if m != ' ' else '[EOW]' for m in mat.columns]
    mat.sort_index(inplace=True)
    ordered_cols = sorted(mat.columns)
    mat = mat.loc[index, ordered_cols].round(4)

    return mat


def view_matrix(matrix):
    fig = px.imshow(matrix, color_continuous_scale='thermal')
    fig.update_layout(
        width=500, height=700, xaxis_title='Next Letter',
        yaxis_title='Current Letter',
        yaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}},
        xaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}}
    )
    fig.update_xaxes(tickangle=0)
    template = "Probability of %{x}<br>after %{y}: %{z}<extra></extra>"
    fig.update_traces(hovertemplate=template)
    st.plotly_chart(fig, use_container_width=True)


def generate_word(mat, start_letter=None):
    if start_letter is None:
        start_letter = np.random.choice(mat.index)
    word = start_letter
    options = mat.columns

    eow = False
    while not eow:
        probs = mat.loc[word[-1], :]
        probs /= sum(probs)
        next_letter = np.random.choice(options, p=probs)
        eow = next_letter == '[EOW]'
        if not eow:
            word += next_letter

    return word


@st.cache_data
def english_probabilities():
    # load words
    nltk.download('words')
    words = nltk.corpus.words.words()
    words = list(set([word.lower() for word in words]))
    words = pd.Series(words).str.replace(r'\W', '', regex=True)

    # calculate probability according to matrix
    mat = st.session_state['English']
    data = []

    for i, word in enumerate(words, 1):
        letters = list(word) + ['[EOW]']
        probs = []
        for n in range(len(letters) - 1):
            prob = mat.loc[letters[n], letters[n+1]]
            probs.append(prob)
        prob = np.product(probs)
        data.append([word, len(word), prob])

    data = pd.DataFrame(data, columns=['word', 'length', 'probability'])
    data = data[data['probability'] > 1e-4]
    return data


languages = ['French', 'Spanish', 'Portuguese', 'German', 'Russian',
             'Italian', 'Malaysian', 'Polish', 'English']

langs = prepare_data()
if languages[-1] not in st.session_state:
    for l, code in zip(languages, langs.columns):
        st.session_state[l] = letter_frequencies(langs[code])

main, tab1, tab2, tab3 = st.tabs(
    ('About', 'Letter Matrix', 'Word Generator', 'Word probabilities')
)

with main:
    st.text('About page')

with tab1:
    selected = st.selectbox(
        'Choose a language to view letter frequencies', [''] + languages
    )
    if selected:
        st.header(f'Letter association matrix for {selected} language')
        view_matrix(st.session_state[selected])

if 'n_valid' not in st.session_state:
    st.session_state['n_valid'] = 0
    st.session_state['n_invalid'] = 0

with tab2.form('Word Generator'):
    language = st.selectbox('Choose a language', [''] + languages)
    letter = st.text_input('Choose a starting letter', max_chars=1)
    submitted = st.form_submit_button("Generate")
    if submitted:
        if language == '':
            st.text('Select a language to generate a word')
        elif letter not in st.session_state[language].index:
            st.text('Invalid letter; try again')
        else:
            word = generate_word(st.session_state[language], letter)
            st.text(f'Generated word:\t{word}')
            code = langs.columns[languages.index(language)]
            if word in langs[code]:
                st.text('Word is valid and in records!')
                st.session_state['n_valid'] += 1
            else:
                st.text('Word is invalid according to records')
                st.session_state['n_invalid'] += 1
            n_valid = st.session_state['n_valid']
            n_invalid = st.session_state['n_invalid']
            st.text(f'{n_valid} valid, {n_invalid} invalid')

with tab3:
    with st.spinner('Calculating word probabilities (this may take a minute, go get some coffee)'):
        probs = english_probabilities()
        probs.sort_values('probability', ascending=False, inplace=True)
    fig = px.box(probs, y='probability', x='length')
    fig.update_layout(
        xaxis_title='Word Length', yaxis_title='Probability',
        yaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}},
        xaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}}
    )
    fig.update_yaxes(type='log')
    st.plotly_chart(fig)
    st.dataframe(probs)

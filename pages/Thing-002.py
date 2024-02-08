#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:27:57 2024

@author: martin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def process_fat_str(s):
    # extracts int and converts ml to gram depending on veg or sunflower oil
    if 'g ' in s or s.endswith('g'):
        num = int(s.split('g')[0])
    elif 'veg. oil' in s:
        num = int(s.split('ml')[0]) * 0.921  # vegetable oil
    else:
        num = int(s.split('ml')[0]) * 0.96  # sunflower oil

    return num


@st.cache_data
def load_data():
    df = pd.read_csv('../data/recipe_data.csv')
    return df


@st.cache_data
def prepare_data(df):
    data = df.copy()

    # format and drop unnecessary columns
    data['flour'] = data['flour'].str.replace('g', '').astype(int)
    data['sugar'] = data['sugar'].str.replace('g', '').astype(int)
    data['fat'] = data['fat'].apply(process_fat_str).astype(int)
    data.drop(columns=['recipe', 'link', 'notes', 'flour_type'], inplace=True)

    # add small value to prevent multiplication/division by zero
    data.iloc[:, 1:] = data.iloc[:, 1:] + 1
    data['fat:sugar ratio'] = data['fat'] / data['sugar']
    data['fat:flour ratio'] = data['fat'] / data['flour']
    data['flour:sugar ratio'] = data['flour'] / data['sugar']

    return data

# %%


def dtree_decision_boundary(cols, dtree=False):
    plot_colors = "wrby"
    classes = data['class'].unique()
    X = data[cols].values
    y = data['class'].values
    global clf
    if dtree:
        clf = DecisionTreeClassifier(
            random_state=42, class_weight='balanced', min_impurity_decrease=5e-2
        )
    else:
        clf = LinearDiscriminantAnalysis(n_components=2)

    clf.fit(X, y)
    score = clf.score(X, y)

    if len(cols) == 2:
        fig, ax = plt.subplots()
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=plt.cm.RdYlBu, response_method="predict",
            xlabel=cols[0], ylabel=cols[1], ax=ax
        )
        for i, color in zip(classes, plot_colors):
            idx = np.where(y == i)
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                c=color,
                label=i,
                cmap=plt.cm.RdYlBu,
                edgecolor="black",
                s=30,
            )
        plt.title(f'Classifier score: {score:.1%}')
        plt.legend()
        plt.tight_layout()

    if isinstance(clf, LinearDiscriminantAnalysis):
        comps = clf.transform(X)
        comps = pd.DataFrame(comps, columns=['LDA 1', 'LDA 2'])
        comps['Class'] = y
        plt.figure()
        sns.scatterplot(x='LDA 1', y='LDA 2', data=comps, hue='Class')
        sns.despine()
        plt.tight_layout()

    return score


# this feature combo gives the best accuracy
dtree_decision_boundary(['fat:flour ratio', 'flour', 'sugar'], False)

# ['fat:flour ratio', 'flour', 'sugar'] for LDA with 3 features: 88%
# ['fat:flour ratio', 'sugar'] for LDA with 2 features: 84.5%
# ['flour:sugar ratio', 'flour', 'fat'] for DT with 3 features: 90%

# %%
df = load_data()
data = prepare_data()

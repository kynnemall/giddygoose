#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:27:57 2024

@author: martin
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def process_fat_str(s):
    """
    Format the fat column of the dataframe for butter and oil

    Parameters
    ----------
    s : string
        amount of butter/fat in the recipe

    Returns
    -------
    num : integer
        grams of butter/oil in the recipe

    """
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
    """
    Load the dataset and use Streamlit caching

    Returns
    -------
    df : pandas dataframe
        recipes dataset

    """
    df = pd.read_csv('data/recipe_data.csv')
    return df


@st.cache_data
def prepare_data(df):
    """
    Format recipes for machine learning. Use Streamlit caching.

    Parameters
    ----------
    df : pandas dataframe
        raw recipes dataset

    Returns
    -------
    data : pandas dataframe
        formatted recipes dataframe

    """
    data = df.copy()

    # format and drop unnecessary columns
    data['flour'] = data['flour'].str.replace('g', '').astype(int)
    data['sugar'] = data['sugar'].str.replace('g', '').astype(int)
    data['fat'] = data['fat'].apply(process_fat_str).astype(int)
    data.drop(columns=['recipe', 'link', 'notes', 'flour_type'], inplace=True)

    # add small value to prevent multiplication/division by zero
    data.iloc[:, 1:] = data.iloc[:, 1:] + 1

    # calculate ingredient ratios
    data['fat:sugar ratio'] = data['fat'] / data['sugar']
    data['fat:flour ratio'] = data['fat'] / data['flour']
    data['flour:sugar ratio'] = data['flour'] / data['sugar']

    return data


@st.cache_resource
def train_model(data, cols):
    """
    Train a Linear Discriminant Analysis (LDA) model

    Parameters
    ----------
    data : pandas dataframe
        formatted recipes dataset
    cols : list(string)
        dataset columns to use in the model

    Returns
    -------
    clf : scikit-learn LDA model
        fitted LDA model

    """
    X = data[cols].values
    y = data['class'].values
    clf = LinearDiscriminantAnalysis(n_components=2)
    clf.fit(X, y)
    return clf


def plot_decision_boundary(data, cols, clf, classes):
    """
    Plot the decision boundary of the trained LDA classifier

    Parameters
    ----------
    data : pandas dataframe
        formatted recipes dataset
    cols : list(string)
        dataset columns to use in the model
    clf : scikit-learn LDA model
        fitted LDA model
    classes : list(string)
        list of classes that can be predicted

    Returns
    -------
    fig : matplotlib figure
        decision boundary plot showing training datapoints

    """
    plot_colors = "wrby"
    X = data[cols].values
    y = data['class'].values
    score = clf.score(X, y)

    # plot the decision boundary
    xlabel = cols[0] if 'ratio' in cols[0] else cols[0] + ' (grams)'
    ylabel = cols[1] if 'ratio' in cols[1] else cols[1] + ' (grams)'
    fig, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=plt.cm.RdYlBu, response_method="predict",
        xlabel=xlabel, ylabel=ylabel, ax=ax
    )

    # plot individual data points
    for i, color in zip(classes, plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0], X[idx, 1], c=color, label=i, cmap=plt.cm.RdYlBu,
            edgecolor="black", s=30,
        )

    plt.title(f'Prediction accuracy: {score:.1%}')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return fig


def plotly_predictions(y_pred, classes):
    """
    Make barplot of prediction class probabilities

    Parameters
    ----------
    y_pred : numpy array
        model prediction probability for each class
    classes : list(string)
        list of classes that can be predicted

    Returns
    -------
    fig : plotly figure
        barplot of 

    """
    fig = px.bar(x=classes, y=y_pred)
    fig.update_layout(
        yaxis_title="Probability", xaxis_title='',
        yaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}},
        xaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}}
    )
    fig.update_traces(hovertemplate="<br>".join(["%{x}", "%{y:.2%}"]))
    return fig


# these feature combos give the best accuracy
# dtree_decision_boundary(['fat:flour ratio', 'flour', 'sugar'], False)
# ['fat:flour ratio', 'flour', 'sugar'] for LDA with 3 features: 88%
# ['fat:flour ratio', 'sugar'] for LDA with 2 features: 84.5%
# ['flour:sugar ratio', 'flour', 'fat'] for DT with 3 features: 90%

features = ['fat:flour ratio', 'sugar']
df = load_data()
data = prepare_data(df)
classes = data['class'].unique()
model = train_model(data, cols=features)
fig = plot_decision_boundary(data, features, model, classes)

# layout
st.markdown("""
    Discussing baked goods recently, I began wondering at what point you
    a biscuit is more like a cake or when pastry is more like bread. It's
    somehwat known that an equal ratio of flour, sugar, and butter makes a
    cake, so maybe you can consider ratios for other things.<br><br>This is
    probably already a baking thing but It's more fun to find out with code
    than just Googling the answer.<br><br>Using Machine Learning and nothing
    but info on quantities of flour, butter, and sugar in a recipe, there now
    exists a model that can classify a baked good as either bread, cake,
    biscuit, or pastry.
""", unsafe_allow_html=True)

st.pyplot(fig)
st.markdown("""
    Use the form below to test your own recipes and see where they fall.
""", unsafe_allow_html=True)

with st.form("Classify your recipe"):
    fat = st.number_input("Amount of butter/fat in grams", min_value=1,
                          max_value=1000, value=1, step=1)
    flour = st.number_input("Amount of flour in grams", min_value=1,
                            max_value=1000, value=1, step=1)
    sugar = st.number_input("Amount of sugar in grams", min_value=1,
                            max_value=1000, value=1, step=1)
    submitted = st.form_submit_button("Classify that recipe!")
    if submitted:
        fat_to_flour = fat / flour
        X_test = np.array([[fat_to_flour, sugar]])
        y_test = model.predict_proba(X_test).ravel()
        plotly_fig = plotly_predictions(y_test, model.classes_)
        st.plotly_chart(plotly_fig, use_container_width=True)

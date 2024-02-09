#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 22:09:24 2024

@author: martin
"""

import numpy as np
import streamlit as st
import plotly.express as px


def simulate(n_women, n_sim, days, cycle_lengths):
    """
    Calculate `n_sim` times the probability of `n_women` having an event
    given a range of `days` and `cycle_lengths`

    Parameters
    ----------
    n_women : integer
        number of women in each simulation population
    n_sim : integer
        number of simulations to perform
    days : integer, list[integer]
        single or list of potential duration in days of a period
    cycle_lengths : integer, list[integer]
        single or list of potential duration in days of a cycle

    Returns
    -------
    all_probs : numpy.narry
        probability of event for each simulation

    """
    #

    low, high = cycle_lengths
    cycle_lengths = np.arange(low, high + 1, dtype=np.float16)
    all_probs = []
    for n in range(n_sim):
        cycles = np.random.choice(cycle_lengths, size=n_women, replace=True)
        if isinstance(days, int):
            probs = days / cycles

        else:
            day_options = np.random.choice(days, size=n_women, replace=True)
            probs = day_options / cycles

        all_probs.append(np.prod(probs))

    all_probs = np.array(all_probs) * 100  # convert to percentage
    return all_probs


# layout
st.markdown("""
    Two friends were wondering what the probability that two women would be 
    on their at the same time. They reasoned it would be 25% assuming the 
    average cycle is 28 days and a period lasts 7 days.<br><br>Being a 
    scientist and mathematically-inclined, I thought it can't be as simple 
    as that;<br>It would actually be about 5.4%.<br><br>The simple thing 
    is to simulate a number of different pairs of women, each with a 
    potential period duration of 5-7 days and cycle length varying from 
    28-35 days. You can vary these parameters yourself and see how the 
    probabilities change.<br><br>NOTE: you can use the Github issues 
    section of this repo to request a change to the limits of these parameters 
    if there's an odd period duration I haven't considered.
""", unsafe_allow_html=True)

with st.form("Probability Calculator"):
    days = st.slider('Period duration (days)', 1, 7, (5, 7), 1)
    d1, d2 = days
    if d1 == d2:
        duration = f'({d1} days)'
    else:
        duration = f'({d1}-{d2} days)'
    cycle_lengths = st.slider('Cycle Length (days)', 28, 35, (28, 35), 1)
    n_women = st.number_input('Number of women in population', 1, 100, 2, 1)
    n_sim = st.slider('Number of simulations', 10, 10000, 100, 10)
    submitted = st.form_submit_button("Run Simulations")
    if submitted:
        days = np.arange(days[0], days[1] + 1)
        c1, c2 = cycle_lengths
        if c1 == c2:
            cycle_str = f'({c1} days)'
        else:
            cycle_str = f'({c1}-{c2} days)'

        probs = simulate(n_women, n_sim, days, cycle_lengths)

        # show mean, min, and max probabilities
        st.header(
            f'Probability that {n_women} women are on their period given cycle length {cycle_str} and period duration {duration}')
        sta, stb, stc = st.columns(3)
        mean_, min_, max_ = probs.mean(), probs.min(), probs.max()
        if min_ >= 0.01:
            pre = 'f'
        else:
            pre = 'e'
        sta.metric('Lowest Probability', f'{probs.min():.2{pre}}%')
        stb.metric('Average Probability', f'{probs.mean():.2{pre}}%')
        stc.metric('Highest Probability', f'{probs.max():.2{pre}}%')

        # plot histogram of probabilities
        bar_color = st.get_option('theme.primaryColor')
        fig = px.histogram(
            probs, nbins=30, template="simple_white",
            color_discrete_sequence=[bar_color]
        )
        fig.update_layout(
            showlegend=False, xaxis_title="% probability", yaxis_title='Count',
            yaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}},
            xaxis={'tickfont': {'size': 16}, 'titlefont': {'size': 20}}
        )
        st.plotly_chart(fig, use_container_width=True)

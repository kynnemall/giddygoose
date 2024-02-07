#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:25:30 2024

@author: martin
"""

import os
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from multiprocessing import Pool

URLS = {
    'Cake': 'https://www.bbcgoodfood.com/recipes/collection/classic-cake-recipes',
    'Biscuit': 'https://www.bbcgoodfood.com/recipes/collection/biscuit-recipes',
    'Pastry': 'https://www.bbcgoodfood.com/search?q=shortcrust+pastry&tab=recipe&mealType=dessert',
}


def get_soup(url):
    response = requests.get(url)
    assert '200' in str(response), f'URL response is not 200: {response}'
    soup = BeautifulSoup(response.content, 'lxml')
    return soup


def extract_recipe_data(url):
    soup = get_soup(url)
    sections = soup.findAll('section')
    for s in sections:
        if s.text.startswith('Ingredients'):
            ingredients = [i.text for i in s.findAll('li')]
            break

    # extract recipe quantities from ingredients
    text_block = '\n'.join(ingredients)
    recipe = pd.DataFrame({
        'recipe': soup.findAll('h1')[0].text,
        'ingredients': text_block,
        'link': url
    }, index=[0])
    return recipe


def get_recipe_data(base_url, group, n_processes):
    soup = get_soup(base_url)
    links = [i['href'] for i in soup.findAll('a', class_='link d-block')][:-3]
    urls = ['https://www.bbcgoodfood.com/' + url for url in links]
    dfs = []

    with Pool(processes=n_processes) as p:
        with tqdm(total=len(urls), desc=group) as pbar:
            for data in p.imap_unordered(extract_recipe_data, urls):
                dfs.append(data)
                pbar.update()
    df = pd.concat(dfs)
    return df


if __name__ == '__main__':
    dfs = []
    n_proc = min(os.cpu_count(), 3)
    print(f'Running multiprocessing with {n_proc} processes')

    for group, url in URLS.items():
        df = get_recipe_data(url, group, n_proc)
        df['class'] = group
        dfs.append(df)

    df = pd.concat(dfs)
    print('\nSaving recipe data to CSV file')
    df.to_csv('../data/recipe_data.csv', index=False)

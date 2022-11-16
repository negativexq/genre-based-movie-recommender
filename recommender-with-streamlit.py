import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


movies = pd.read_csv('datasets/movies.csv')
ratings = pd.read_csv('datasets/ratings.csv')
df = pd.merge(ratings, movies, how='left', on='movieId')


# the function to extract titles
def extract_title(title):
    year = title[len(title) - 5:len(title) - 1]

    # some movies do not have the info about year in the column title. So, we should take care of the case as well.

    if year.isnumeric():
        title_no_year = title[:len(title) - 7]
        return title_no_year
    else:
        return title


# the function to extract years
def extract_year(title):
    year = title[len(title) - 5:len(title) - 1]
    # some movies do not have the info about year in the column title. So, we should take care of the case as well.
    if year.isnumeric():
        return int(year)
    else:
        return np.nan


# change the column name from title to title_year
movies.rename(columns={'title': 'title_year'}, inplace=True)
# remove leading and ending whitespaces in title_year
movies['title_year'] = movies['title_year'].apply(lambda x: x.strip())
# create the columns for title and year
movies['title'] = movies['title_year'].apply(extract_title)
movies['year'] = movies['title_year'].apply(extract_year)

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4)
                                             for c in combinations(s.split('|'), r=i)))
tfidf_matrix = tf.fit_transform(movies['genres'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])


def movie_recommendations(i):
    """
    i : Movie

    """
    ix = cosine_sim_df.loc[:, i].to_numpy().argpartition(range(-1, -10, -1))
    closest = cosine_sim_df.columns[ix[-1:-(10 + 2):-1]]
    closest = closest.drop(i, errors='ignore')
    return pd.DataFrame(closest).merge(movies[['title', 'genres']]).head(10)

st.title('Genre based movie recommendation system')
text_input = st.text_input("Enter a movie name","")

result = st.button(" Get recommendation" )

if result:
    st.write(movie_recommendations(text_input))




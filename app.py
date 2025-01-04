# import pickle
# import streamlit as st
# import requests

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=6f1823c4ab532a80d83b7cc68de97cd2&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path




# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     recommended_movie_names = []
#     recommended_movie_posters = []
#     for i in distances[1:6]:
#         # fetch the movie poster
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_posters.append(fetch_poster(movie_id))
#         recommended_movie_names.append(movies.iloc[i[0]].title)

#     return recommended_movie_names,recommended_movie_posters


# st.header('Movie Recommender System')
# movies = pickle.load(open('movie_list.pkl','rb'))
# similarity = pickle.load(open('similarity.pkl','rb'))

# movie_list = movies['title'].values
# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )

# if st.button('Show Recommendation'):
#     recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    
#     col1, col2, col3, col4, col5 = st.columns(5)

#     with col1:
#         st.text(recommended_movie_names[0])
#         st.image(recommended_movie_posters[0])
#     with col2:
#         st.text(recommended_movie_names[1])
#         st.image(recommended_movie_posters[1])

#     with col3:
#         st.text(recommended_movie_names[2])
#         st.image(recommended_movie_posters[2])
#     with col4:
#         st.text(recommended_movie_names[3])
#         st.image(recommended_movie_posters[3])
#     with col5:
#         st.text(recommended_movie_names[4])
#         st.image(recommended_movie_posters[4])

import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import PorterStemmer
import streamlit as st
import requests

# Fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=6f1823c4ab532a80d83b7cc68de97cd2&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', "")
    return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else ""

# Preprocessing functions
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except Exception:
        return []

def collapse(L):
    return [i.replace(" ", "").lower() for i in L]

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])

# Load and preprocess data
movies = pd.read_csv(r"tmdb_5000_movies.csv", encoding='ISO-8859-1', sep=',',   engine='python', on_bad_lines='skip')
credits = pd.read_csv(r"tmdb_5000_credits.csv", encoding='ISO-8859-1', sep=',',  engine='python',  on_bad_lines='skip')

movies.columns = movies.columns.str.strip()
credits.columns = credits.columns.str.strip()

# Verify 'title' column exists
if 'title' not in movies.columns:
    print("Error: 'title' column is missing in movies DataFrame.")
    print("Available columns in movies:", list(movies.columns))
if 'title' not in credits.columns:
    print("Error: 'title' column is missing in credits DataFrame.")
    print("Available columns in credits:", list(credits.columns))

# Proceed only if 'title' exists in both
if 'title' in movies.columns and 'title' in credits.columns:
    # Merge DataFrames
    merged_data = movies.merge(credits, on='title', how='inner')
    print("Merged DataFrame:", merged_data.head())
else:
    print("Merge operation skipped due to missing 'title' column.")
#movies = movies.merge(credits, on='title')
required_columns = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
missing_columns = [col for col in required_columns if col not in movies.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    movies = movies[required_columns]

#j

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert).apply(collapse)
movies['keywords'] = movies['keywords'].apply(convert).apply(collapse)

movies['cast'] = movies['cast'].apply(convert).apply(lambda x: x[:3]).apply(collapse)
movies['crew'] = movies['crew'].apply(
    lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director']
).apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())
movies['tags'] = movies['tags'].apply(stem)

# Feature extraction and similarity computation
cv = CountVectorizer(max_features=3000, stop_words='english')  # Reduced max_features
vector = cv.fit_transform(movies['tags'])
similarity = linear_kernel(vector)

# Recommendation function
def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movie_names.append(movies.iloc[i[0]].title)
            recommended_movie_posters.append(fetch_poster(movie_id))
        return recommended_movie_names, recommended_movie_posters
    except Exception:
        return [], []

# Streamlit UI
st.header('Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    if recommended_movie_names:
        cols = st.columns(5)
        for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
            with col:
                st.text(name)
                st.image(poster)
    else:
        st.text("No recommendations available.")

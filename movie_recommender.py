import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "movies.csv"
movies_data = pd.read_csv(DATA_PATH)
print("Dataset shape:", movies_data.shape)
print("Columns:", movies_data.columns.tolist())

selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")
combined_features = (movies_data["genres"] + " " + movies_data["keywords"] + " " +
                     movies_data["tagline"] + " " + movies_data["cast"] + " " +
                     movies_data["director"])

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print("TF-IDF matrix:", feature_vectors.shape)
similarity = cosine_similarity(feature_vectors)
print("Similarity matrix:", similarity.shape)

def recommend_movies(movie_name, n=10):
    titles = movies_data["title"].tolist()
    matches = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.35)
    if not matches:
        return [], None
    close_match = matches[0]
    index = movies_data.index[movies_data.title == close_match][0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    results=[]
    for idx, score in scores[1:]:
        results.append((movies_data.iloc[idx]["title"], float(score)))
        if len(results) >= n: break
    return results, close_match

query = "Avatar"
recommendations, matched = recommend_movies(query, 10)
print(f"Input: {query} | Matched title: {matched}")
for i,(title,score) in enumerate(recommendations,1): print(f"{i}. {title} (similarity={score:.3f})")

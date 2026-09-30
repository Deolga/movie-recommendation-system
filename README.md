# Movie Recommendation System

## Overview
A content-based movie recommendation system that suggests movies similar to a user's chosen title. It uses movie metadata (genres, keywords, tagline, cast, and director), converts the combined text into TF-IDF vectors, and ranks movies using cosine similarity.

## Dataset
`movies.csv` is the supplied internship dataset.

## Methodology
1. Load and inspect the movie dataset.
2. Select descriptive metadata fields.
3. Replace missing metadata with empty strings.
4. Combine the selected fields into one text representation.
5. Convert text to TF-IDF vectors.
6. Calculate pairwise cosine similarity.
7. Use fuzzy title matching with `difflib` so small spelling differences can be handled.
8. Return the top similar movies.

## Example
For the demonstration query `Avatar`, the notebook prints the matched title and ten ranked recommendations with similarity scores.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Movie_Recommendation_System.ipynb
```

## Limitations
This is content-based rather than collaborative filtering. Recommendations depend on the available metadata and do not explicitly learn from user ratings or viewing history.

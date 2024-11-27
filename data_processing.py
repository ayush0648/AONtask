import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Step 1: Load Datasets
movies = pd.read_csv('movie.csv')
ratings = pd.read_csv('rating.csv')
genome_scores = pd.read_csv('genome_scores.csv')
genome_tags = pd.read_csv('genome_tags.csv')

# Step 2: Clean and Process Genres
# Split genres into lists
movies['genres'] = movies['genres'].str.split('|')

# Embed genres using TF-IDF to capture relationships
tfidf = TfidfVectorizer()
genre_text = movies['genres'].apply(lambda x: ' '.join(x)).tolist()
tfidf_matrix = tfidf.fit_transform(genre_text)

# Reduce dimensionality with SVD to create embeddings
svd = TruncatedSVD(n_components=10, random_state=42)  # 10-dimensional genre embeddings
genre_embeddings = svd.fit_transform(tfidf_matrix)
movies = pd.concat([movies, pd.DataFrame(genre_embeddings, columns=[f'genre_emb_{i}' for i in range(10)])], axis=1)

# Step 3: Balance Genres
# Assign movies to "Drama," "Comedy," or "Other" for balancing
movies['main_genre'] = movies['genres'].apply(lambda g: 'Drama' if 'Drama' in g else 'Comedy' if 'Comedy' in g else 'Other')

# Downsample Drama and oversample other genres to balance
drama = movies[movies['main_genre'] == 'Drama'].sample(frac=0.47, random_state=42)
comedy = movies[movies['main_genre'] == 'Comedy']
other = movies[movies['main_genre'] == 'Other'].sample(len(comedy), random_state=42)

balanced_movies = pd.concat([drama, comedy, other], ignore_index=True)

# Step 4: Filter Low-Rated Movies and Rare Users
# Remove movies with less than 50 ratings
movie_ratings = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
movie_ratings.columns = ['average_rating', 'rating_count']
popular_movies = movie_ratings[movie_ratings['rating_count'] > 50].index
balanced_movies = balanced_movies[balanced_movies['movieId'].isin(popular_movies)]

# Remove users with fewer than 10 ratings
user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts > 10].index
filtered_ratings = ratings[ratings['userId'].isin(active_users)]

# Merge ratings with balanced movies
filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(balanced_movies['movieId'])]

# Normalize ratings
scaler = MinMaxScaler()
filtered_ratings['rating'] = scaler.fit_transform(filtered_ratings[['rating']])

# Step 5: Process Genome Data (Tags)
# Merge genome scores with tags
genome_data = pd.merge(genome_scores, genome_tags, on='tagId')

# Apply SVD to the genome scores to create embeddings
tag_matrix = genome_data.pivot(index='movieId', columns='tag', values='relevance').fillna(0)
svd_tags = TruncatedSVD(n_components=10, random_state=42)  # 10-dimensional tag embeddings
tag_embeddings = svd_tags.fit_transform(tag_matrix)

# Add tag embeddings to movies dataset
tag_embeddings_df = pd.DataFrame(tag_embeddings, columns=[f'tag_emb_{i}' for i in range(10)], index=tag_matrix.index)

# Reset the index to make 'movieId' a column, not an index
tag_embeddings_df = tag_embeddings_df.reset_index()

# Merge the tag embeddings with the movie dataset
final_movies = pd.merge(balanced_movies, tag_embeddings_df, on='movieId', how='left')

# Step 6: Save Cleaned Data for Modeling
final_movies.to_csv('final_movies_with_embeddings.csv', index=False)
filtered_ratings.to_csv('final_filtered_ratings.csv', index=False)

print("Data processing completed. Cleaned data saved to 'final_movies_with_embeddings.csv' and 'final_filtered_ratings.csv'.")

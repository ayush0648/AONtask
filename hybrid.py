import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('final_movies_with_embeddings.csv')
ratings = pd.read_csv('final_filtered_ratings.csv')

genre_embeddings = movies[[f'genre_emb_{i}' for i in range(10)]].values
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(genre_embeddings)

rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
knn_user = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_user.fit(rating_matrix)

def hybrid_recommendations(user_id, movie_id, n_recommendations=5, content_weight=0.8, collab_weight=0.2):
    # Collaborative Filtering: Find similar users
    user_index = ratings[ratings['userId'] == user_id].index[0]
    _, similar_users = knn_user.kneighbors(rating_matrix.iloc[user_index].values.reshape(1, -1),
                                           n_neighbors=n_recommendations + 1)

    # Content-based Filtering: Find similar movies based on genre embeddings
    movie_index = movies[movies['movieId'] == movie_id].index[0]
    _, similar_movies = knn.kneighbors(genre_embeddings[movie_index].reshape(1, -1), n_neighbors=n_recommendations + 1)

    # Combine recommendations from both
    similar_user_ids = similar_users.flatten()[1:]  # Exclude the user themselves
    similar_movie_indices = similar_movies.flatten()[1:]  # Exclude the movie itself

    # Collaborative Filtering Recommendations: Movies liked by similar users
    user_movie_ids = ratings[ratings['userId'].isin(similar_user_ids)]['movieId'].unique()

    # Combine and deduplicate recommendations
    all_movie_ids = set(similar_movie_indices) | set(user_movie_ids)
    all_movie_indices = [movies[movies['movieId'] == mid].index[0] for mid in all_movie_ids if
                         mid in movies['movieId'].values]

    # Limit to top-N recommendations
    all_movie_indices = all_movie_indices[:n_recommendations]

    # Compute content-based similarity scores
    content_similarity_scores = cosine_similarity(
        genre_embeddings[all_movie_indices],
        genre_embeddings[movie_index].reshape(1, -1)
    ).flatten()

    # Compute collaborative filtering similarity scores (based on movie rating similarity)
    collab_similarity_scores = cosine_similarity(
        rating_matrix.iloc[similar_users.flatten()[1:]].values,
        rating_matrix.iloc[movies[movies['movieId'] == movie_id].index[0]].values.reshape(1, -1)
    ).flatten()

    # Combine content-based and collaborative filtering scores
    final_scores = (content_weight * content_similarity_scores) + (collab_weight * collab_similarity_scores)

    # Build final recommendations
    recommended_movies = movies.iloc[all_movie_indices].copy()
    recommended_movies['similarity_score'] = final_scores

    # Sort by the final similarity score (descending)
    recommended_movies = recommended_movies.sort_values(by='similarity_score', ascending=False)

    return recommended_movies[['movieId', 'title']]


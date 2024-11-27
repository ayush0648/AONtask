import pandas as pd
from hybrid import hybrid_recommendations
from rnn import recommend_next_movie
from datetime import datetime

movies = pd.read_csv('final_movies_with_embeddings.csv')
ratings = pd.read_csv('final_filtered_ratings.csv')

movie_id_watched = int(input("Enter the movieId of the movie you watched: "))
user_id = int(input("Enter your userId: "))

# Ask user for the model type to run
model_type = input("Enter model type ('hybrid' or 'rnn'): ").strip().lower()

# Get recommendations based on the selected model
if model_type == 'hybrid':
    recommendations = hybrid_recommendations(user_id, movie_id_watched, n_recommendations=5)
elif model_type == 'rnn':
    recommendations = recommend_next_movie(user_id, movie_id_watched, n_recommendations=5)
else:
    raise ValueError("Invalid model type. Choose either 'hybrid' or 'rnn'.")

# Display recommendations
print(f"\nTop 5 movie recommendations based on movieId {movie_id_watched} using the {model_type} model:")
print(recommendations)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"\nTime of recommendation: {current_time}")

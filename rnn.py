import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

movies = pd.read_csv('final_movies_with_embeddings.csv')
ratings = pd.read_csv('final_filtered_ratings.csv')

movie_encoder = LabelEncoder()
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

# Restrict sequences to the last 20 movies per user
sequence_length = 20
user_movie_sequences = ratings.groupby('userId')['movieId'].apply(
    lambda x: x.tolist()[-sequence_length:]
).values

# Pad sequences to a fixed length
padded_sequences = pad_sequences(user_movie_sequences, maxlen=sequence_length, padding='post')

model = Sequential()
model.add(Embedding(input_dim=len(movie_encoder.classes_), output_dim=100, input_length=sequence_length - 1))
model.add(LSTM(128, return_sequences=False))  # Increased LSTM units for better representation
model.add(Dropout(0.4))  # Increased dropout for regularization
model.add(Dense(len(movie_encoder.classes_), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X = padded_sequences[:, :-1]  # Input: All except the last movie in each sequence
y = padded_sequences[:, -1]   # Target: The last movie in each sequence

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_val, y_val), verbose=1)

def recommend_next_movie(user_id, movie_id_watched, n_recommendations=5):
    # Prepare user sequence and predict next movies
    user_sequence = ratings[ratings['userId'] == user_id]['movieId'].values.tolist()
    user_sequence = user_sequence[-(sequence_length - 1):]  # Use the last `sequence_length - 1` movies
    user_sequence_padded = pad_sequences([user_sequence], maxlen=sequence_length - 1, padding='post')

    predicted_scores = model.predict(user_sequence_padded)[0]
    predicted_movie_indices = np.argsort(predicted_scores)[-n_recommendations:][::-1]
    recommended_movie_ids = movie_encoder.inverse_transform(predicted_movie_indices)

    # Get recommended movies with similarity scores
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    recommended_movies['similarity_score'] = predicted_scores[predicted_movie_indices]

    return recommended_movies[['movieId', 'title']]  # Exclude similarity score for cleaner output

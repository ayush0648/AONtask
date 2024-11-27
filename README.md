# AONtask
Ayushmant Bharadwaj's submission for the AON Data Scientist interview assignment. The task is to create a movie recommender system using the [Movie Lens dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download). Here's a step-by-step approach to the system:

## Data Preprocessing

1. **Data Balancing:** The data is unbalanced as Drama is 17%, Comedy is 8%, and the rest 75% falls under other genres. To balance this:
+ Genres have been embedded using TF-IDF and I have used SVD to reduce dimensionality, and to find relationships between genres.
+ Drama has been undersampled and other genres have been oversampled to balance the dataset.
2. **Data Cleaning:** The first model I build will use a hybrid of content-based and collaborative filtering. To improve the quality of recommendations, I will remove movies which have received less than 50 ratings, and users who have given less than 10 ratings.
3. **Genome Tags:** The remaining dataset will then be normalized. Furthermore, genome tags will be extracted to understand the relevance scores and enrich the data.

## Methodologies Used:

I have used 2 approaches to the problem: _Hybrid Filtering_ and _Recurrent Neural Networks (RNN)_.
1. **Hybrid Filtering:** In this approach, I used KNN clustering to cluster the data based on 2 filtering types: _content-based_ and _collaborative_ filtering. The recommendations gathered from these 2 clusters are then combined to give a single list using a weighted average which can be modified, depending on which form of filtering is preferred. The variables which the recommendations are based upon are:
    + Genre of the movie watched.
    + Movies that similar users like to watch.
    + User's Watching history.
2. **Recurrent Neural Network:** RNNs can predict the next movie a user might be interested in watching based on their viewing history. The model processes a sequence of movies watched by the user, learning patterns in their preferences to recommend the most likely next movie in the sequence.

**Recommended Deployment:** 
1. Hybrid Filtering model is suggested as it would be more adaptable to our case study. This case uses both user ratings and movie attributes, hence this model will provide better recommendations and more diversity in suggestions.
2. It provides more control over the recommendation system as you can influence each component's control over the model (content vs. collaborative).
3. As it doesn't require the time consuming training process than an RNN requires, it is more suitable and efficient for realtime applications.

This leaves us with 2 remaining questions:
1. Suppose, a user has watched a particular movie ‘X’ and is interested in watching another movie ‘Y’, which may or may not be to their immediate taste. So, can I recommend a few movies in sequence, that this user can watch to ease themselves into watching ‘Y'.
2. Using ‘Exploration vs Exploitation’ strategies, can I suggest ways in which you can get an user to try out different niches.

These 2 issues can be solved in the methodologies used in the following ways:
1. The 1st problem is a sequential recommendation problem. The _RNN Model_ is naturally good for this task due to its ability to model sequential behavior, including understanding how a user might progress from one genre or type of movie to another.
2. Both models are good at understanding exploitation. Although, the _Hybrid Model_ gives you more control over exploration by adjusting how similar movies are to the ones the user has already watched and can incorporate randomness to encourage niche discovery.


## Requirements

You should have the following Python dependencies installed:

- `pandas`
- `numpy`
- `keras`
- `tensorflow`
- `scikit-learn`

You can install all the dependencies by running the following command:

pip install -r requirements.txt

## How to Run

1. Clone the repository locally.
2. Use the Kaggle link provided to download the dataset in the same folder.
3. Run data_processing.py to create the clean datasets. You can do that by using the command:

python data_processing.py

4. Once the .csv files are made, run the file run.py. You can do that by using the command:

python run.py

## File Structure
1. run.py: Main script to run the hybrid and RNN recommendation models.
2. hybrid.py: Contains the hybrid recommendation model implementation.
3. rnn.py: Contains the RNN model implementation for sequential recommendations.
4. final_movies_with_embeddings.csv: Dataset containing movie embeddings.
5. final_filtered_ratings.csv: Dataset containing user ratings. 

## Future Works:

1. The RNN model requires a lot of work. As it was a huge dataset, the model has been adjusted to give the highest possible efficiency by sacrificing accuracy.
2. We can use diversity filters to make users explore more niches, and tweak the influence of collaborative filtering to influence the recommendations based on users similar to them.

### Please contact me at ayushmant.bharadwaj@gmail.com or +91 9958493981 for any doubts! 

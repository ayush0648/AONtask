# AONtask
My submission for the AON Data Scientist interview assignment. The task is to create a movie recommender system using the [Movie Lens dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download). Here's a step-by-step approach to the system:

## Data Preprocessing

1. **Data Balancing:** The data is unbalanced as Drama is 17%, Comedy is 8%, and the rest 75% falls under other genres. To balance this:
+ Genres have been embedded using TF-IDF and I have used SVD to reduce dimensionality, and to find relationships between genres.
+ Drama and other genres have been undersampled to balance the dataset.
2. **Data Cleaning:** The first model I build will use a hybrid of content-based and collaborative filtering. To improve the quality of recommendations, I will remove movies with less than 50 ratings and users who have given less than 10 ratings.
3. **Genome Tags:** The remaining dataset will then be normalized. Furthermore, genome tags will be extracted to understand the relevance scores and enrich the data.

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

##Future Works:

1. The RNN model requires a lot of work. As it was a huge dataset, the model has been adjusted to give the highest possible efficiency by sacrificing accuracy.
2. We can use diversity filters to make users explore more niches, and tweak the influence of collaborative filtering to influence the recommendations based on users similar to them.

### Please contact me at ayushmant.bharadwaj@gmail.com or +91 9958493981 for any doubts! 

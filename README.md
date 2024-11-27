# AONtask
My submission for the AON Data Scientist interview assignment. The task is to create a movie recommender system using the [Movie Lens dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download). Here's a step-by-step approach to the system:

## Data Preprocessing

1. **Data Balancing**: The data is unbalanced as Drama is 17%, Comedy is 8%, and the rest 75% falls under other genres. To balance this-
                      - Genres have been embedded using TF-IDF and I have used SVD to reduce dimensionality, and to find relationships between genres.
                      - Drama and other genres have been undersampled to balance the dataset.
   The first model I build will use a hybrid of content-based and collaborative filtering. To improve the quality of recommendations, I will remove movies with less than 50 ratings and users who have given less than 10 ratings.
   The remaining dataset will then be normalized. Furthermore, genome tags will be extracted to understand the relevance scores and enrich the data.
   I will use Elastic Search on the dataset to make querying more efficient.


# This is the repository for the CBD project of the course 23/24. Subgroup number 24.

It has been tested with Python 3.8.10 and Python 3.12.3

To install libraries with pip:
pip install -r requirements.txt

To install the nltk necessary files (In a python console):
import nltk
nltk.download('stopwords')
nltk.download('punkt')

Sources for the datasets uploaded here in the repository.

[movie_genres](https://www.kaggle.com/datasets/programmarself/movies-genres/data)

[movies_metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)

## Changes made to create each saved tree (Some code to the ):
-----------------------------------------------------------------
0 and 1: defaults
2: max_depth = 400, min_samples_split = 200, random_state = 20.
3: max_depth = 400, min_samples_split = 200, random_state = 20. Less data of the most common genres
4: max_depth = 200, min_samples_split = 200, random_state = 20.
5: max_depth = 200, min_samples_split = 1000, random_state = 20.
6: max_depth = 200, min_samples_split = 1000, random_state = 20. Even Less data of the most common genres.
7: max_depth = 200, min_samples_split = 1000, random_state = 20. All the data. Added stemming. Limiting tf-idf vocabulary to 20k
8: Limit tf-idf to 50k
9: Removed tf-idf limiting
10: max_depth = 100, min_samples_split = 1000, random_state = 20
11: max_depth = 50, min_samples_split = 1000, random_state = 20


import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import string
import nltk
import requests
import pickle 

known_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
ntlk_stopwords = set(stopwords.words('english'))

# Code to download the file
# stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
# stopwords = set(stopwords_list.decode().splitlines())
with open(os.path.abspath(os.path.dirname(__file__)) + os.sep + "stopwords.pkl", "rb") as f:
    extra_stopwords = pickle.load(f)

# Check if stopwords are downloaded
try:
    nltk.corpus.stopwords.words('english')
except Exception:
    nltk.download('stopwords')

# Check if punkt tokenizer is downloaded
try:
    nltk.tokenize.word_tokenize("Test")
except Exception:
    nltk.download('punkt')

"""
Encodes a list of genres to a binary formated string for easier multigenre movies handling.
"""
def encode_genres(genres):
    res = ["0"] * 20
    for genre in genres:
        if genre in known_genres:
            # Create a bytes string with 20 bits, 1 for each genre and set to 1 the corresponding bit.
            res[known_genres.index(genre)] = "1"

    return ''.join(res)

def decode_genres(string):
    genres = []
    for index in range(0, len(string)):
        if string[index] == "1":
            genres.append(known_genres[index])

    return genres

"""
Uses the nltk library to tokenize and clean the movie info.
"""
def clean_text(text):
    # Tokenizes the text by words
    tokens = word_tokenize(text)

    # Remove apostrophes, grave accents and triple dots
    action_function = lambda word: word.replace("'s", "").replace("'", "").replace("`", "").replace("...", "")
    tokens = [action_function(word) for word in tokens]
    
    """
    Removes punctuation
    Removes basic stop words (a, the, of, on...)
    Extra Stopwords file taken from this thread: https://gist.github.com/sebleier/554280
    """
    filter_function = lambda word: word not in string.punctuation and word.lower() not in ntlk_stopwords # and word.lower() not in extra_stopwords
    tokens = [word for word in tokens if filter_function(word)]
    
    # Stemming to get words in base form. Set to remove duplicates
    stemmer = PorterStemmer()
    tokens = list(set([stemmer.stem(word) for word in tokens]))

    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


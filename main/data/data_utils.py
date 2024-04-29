
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import string

known_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

ntlk_stopwords = set(stopwords.words('english'))

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
    filter_function = lambda word: word not in string.punctuation and word.lower() not in ntlk_stopwords
    tokens = [word for word in tokens if filter_function(word)]
    
    # Stemming to get words in base form. Set to remove duplicates
    stemmer = PorterStemmer()
    tokens = list(set([stemmer.stem(word) for word in tokens]))

    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


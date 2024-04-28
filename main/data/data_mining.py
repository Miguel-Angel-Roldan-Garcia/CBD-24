
import csv
import os
import ast
import pickle
import random
import time

from collections import namedtuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from main.data.data_utils import clean_text, decode_genres, encode_genres

dataset_folder = os.path.abspath(os.path.dirname(__file__)) + os.sep + "dataset"
processed_data_folder = os.path.abspath(os.path.dirname(__file__)) + os.sep + "processedData" 
trees_folder = os.path.abspath(os.path.dirname(__file__)) + os.sep + "trees"

tree_file = lambda: trees_folder + os.sep + f"tree_{int(len(os.listdir(trees_folder))/2)}.pkl"
tree_result_file = lambda: trees_folder + os.sep + f"tree_result_{int(len(os.listdir(trees_folder))/2)}.txt"
processed_data_file = processed_data_folder + os.sep + "data.pkl"

if not os.path.exists(processed_data_folder):
    os.mkdir(processed_data_folder)

if not os.path.exists(trees_folder):
    os.mkdir(trees_folder)

def data_mine():
    # Step 1: Load the datasets
    data = []

    if os.path.exists(processed_data_file) and os.path.getsize(processed_data_file) > 0:
        with open(processed_data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(dataset_folder + os.sep + "movies_metadata.csv", encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter = ",", )
            headers = next(reader)

            Metadata = namedtuple("Metadata", [i for i in headers])
            UsefulData = namedtuple("Data", ["genres", "original_title", "overview", "production_companies"])

            for line in reader:
                # A few lines have less values than required by the header. We choose to ignore these as they are an insignificant number.
                if len(line) != len(headers):
                    continue
                
                # Step 1.1 For efficiency, we clean the data right here
                md = Metadata(*line)

                # Remove all the columns which are not necessary
                useful_data = UsefulData(md.genres, md.original_title, md.overview, md.production_companies)

                # If any property is empty, the line is ignored
                empties = False
                for property in useful_data:
                    if property == None or property == "":
                        empties = True
                        break
                
                if empties:
                    continue

                """
                All properties are either strings or string representations of python objects.
                Here we parse each and take whatever data is useful. For example, we ignore ids.
                The string representations can be parsed using the ast module. The literal strings
                can be ignored.
                """
                properties = []
                for property in useful_data:
                    if property.startswith("[") or property.startswith("{"):
                        try:
                            property = ast.literal_eval(property)
                        except Exception:
                            # The property is a normal string that starts with "[" or "{" and should be ignored.
                            properties.append(property)
                            continue

                        if isinstance(property, list):
                            property = [i["name"] for i in property]
                        else:
                            property = property["name"]

                    properties.append(property)

                useful_data = UsefulData(*properties)

                # Join all the properties into a single string
                info = ""
                for property in useful_data[1:]:
                    if isinstance(property, list):
                        property = ' '.join(property)
                    
                    info += property + " "
                info = info.strip()

                # Now that all the info is in a single string, we can use nltk to clean it properly
                clean_info = clean_text(info)

                for genre in useful_data.genres:
                    data.append((genre, clean_info))

        # Now we repeat the process with a second dataset

        with open(dataset_folder + os.sep + "movies_genres.csv", encoding = "utf-8") as f:
            reader = csv.reader(f, delimiter = "	")
            headers = next(reader)

            # We remove the title and plot headers to leave just the genres
            headers = headers[2:]

            for line in reader:
                # We remove only the title
                line = list(line[1:])

                # If any property is empty, the line is ignored
                empties = False
                for property in line:
                    if property == None or property == "":
                        empties = True
                        break
                
                if empties:
                    continue

                # We can use nltk to clean the plot info
                info = line.pop(0)
                clean_info = clean_text(info)

                for genre_index in range(0, len(line)):
                    genre_bin = line[genre_index]
                    if genre_bin == "1":
                        genre = headers[genre_index]
                        data.append((genre, clean_info))       

        with open(processed_data_file, 'wb') as f:
            pickle.dump(data, f)

    return data


def train_tree():
    data = data_mine()

    """
    We count the number of rows of each genre so that we can eliminate those with few rows.
    This is done because these genres will not have enough information for the classifier
    to learn meaningful patterns and so will only introduce noise into the model.
    We will be discarding those genres with less than 3% of the total rows.
    """
    count = {}
    for i in data:
        i = i[0]
        if(i in count.keys()):
            count[i] += 1
        else:
            count[i] = 1

    total_rows = len(data)
    threshold = total_rows * 0.03
    filtered_genres = [k for k,v in count.items() if v >= threshold]
    data = [d for d in data if d[0] in filtered_genres]
    start_time = time.time()

    # Step 2: Extract the features of each movie's info. We will be using the TF-IDF approach.
    # Step 2.1: Separate data into training and evaluation sets
    X = [item[1] for item in data]
    y = [item[0] for item in data]
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=20)
    print("Data splitted")
    
    # Step 2.2: Feature extraction with TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    print("x train fit transformed")
    X_eval_tfidf = tfidf_vectorizer.transform(X_eval)
    print("x eval transformed")
    
    # Step 2.3: Train Decision Tree Classifier
    """
    Changes made to create each saved tree:
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
    """
    decision_tree = DecisionTreeClassifier(max_depth = 100, min_samples_split = 1000, random_state = 20)
    decision_tree.fit(X_train_tfidf, y_train)
    print("Decision tree trained")

    # Saving the vectorizer and tree to classify later
    with open(tree_file(), 'wb') as f:
        pickle.dump((tfidf_vectorizer, decision_tree), f)

    # Step 2.4: Evaluate the Model
    y_pred = decision_tree.predict(X_eval_tfidf)

    with open(tree_result_file(), "w") as f:
        f.write(classification_report(y_eval, y_pred))

    print(f"Time taken to build the tree: {time.time() - start_time}")

def classify(description):
    clean_description = clean_text(description)

    with open(trees_folder + os.sep + "tree_5.pkl", 'rb') as f:
        tfidf_vectorizer, decision_tree = pickle.load(f)

    X_eval_tfidf = tfidf_vectorizer.transform([clean_description])

    y_pred = decision_tree.predict(X_eval_tfidf)
    return y_pred

    
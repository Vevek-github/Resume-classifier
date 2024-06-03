# Ensure necessary NLTK data packages are downloaded
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('conll2000')
nltk.download('brown')

# Import libraries
from extract_content import extract_files_to_csv
import re
import string
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import warnings
#import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from textblob import TextBlob
#from textblob.sentiments import NaiveBayesAnalyzer
#from textblob.np_extractors import ConllExtractor
#from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define stopwords
stop_words = set(stopwords.words('english'))


def resume_classifier(folder_path : str)->pd.DataFrame :

    # Define the directory containing the test files
    test_directory = folder_path
    output_csv_path = r"Resume content\test_resume.csv"
    from extract_content import extract_files_to_csv
    # Call the function to process the test directory and save to CSV
    extract_files_to_csv(test_directory, output_csv_path)

    #test_file_path  = r"C:\Users\vevek\OneDrive\Desktop\Python\Resume classifier\Testing Resumes"
    df = pd.read_csv(r"Resume content\test_resume.csv")


    print(df)

    #################################################### Cleaning the data ################################################

    df['Resumes'] = df['Resumes'].str.lower()   # Lower case 

    tokens = df.Resumes.map(nltk.word_tokenize)
    df['Tokenized_Resumes'] = tokens

    # Function to remove punctuation, emojis, and SVG-type characters from a list of tokens
    def clean_tokens(tokens):
                                                                            # Create a translation table for punctuation
        table = str.maketrans('', '', string.punctuation)
        cleaned_tokens = []
        for token in tokens:
                                                                            # Remove punctuation
            token = token.translate(table)
                                                                            # Remove emojis and SVG-type characters
            token = re.sub(r'[^\w\s,]', '', token, flags=re.UNICODE)
                                                                            # Remove stopwords  and not empty
            if token and token not in stop_words:
                cleaned_tokens.append(token)
        return cleaned_tokens
                                                                            # Apply the function to the 'Tokenized_Resumes' column
    df['Cleaned_Tokenized_Resumes'] = df['Tokenized_Resumes'].apply(clean_tokens)

                                                                            # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
                                                                            # Function to apply stemming and lemmatization
    def stem_and_lemmatize(tokens):
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
        return lemmatized_tokens
                                                                            # Apply stemming and lemmatization
    df['Cleaned_Tokenized_Resumes'] = df['Cleaned_Tokenized_Resumes'].apply(stem_and_lemmatize)

    df['Absolute_clean_resume'] = df['Cleaned_Tokenized_Resumes'].apply(lambda tokens: " ".join(tokens))


    #print(df)

    ############################################## Model ########################

    requiredText = df['Absolute_clean_resume'].values

    import joblib

    loaded_vectorizer = joblib.load(r"Model\tfidf_vectorizer.joblib")
    WordFeatures = loaded_vectorizer.transform(requiredText)

    #print(WordFeatures)

    loaded_LG = joblib.load(r"Model\weighted_logistic.joblib")
    prediction = loaded_LG.predict(WordFeatures)
    print(prediction)
    df["Prediction"] = prediction
    return df

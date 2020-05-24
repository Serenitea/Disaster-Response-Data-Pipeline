# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys

#machine learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats.mstats import gmean

#NLP
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def load_data(database_filepath):
    '''
    load data from database
    my db filepath: 'sqlite:///disaster_response.db'
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(table_name='message_categories', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y


def tokenize(text):
    '''
    IN: 
        raw text for tokenizing via the following steps: 
            - normalized, punctuation removed, stop words removed, stemmed, and lemmatized
    OUT:
        tokenized text
    '''
    #Normalize text and remove punctuation
    normalized_txt = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    words = word_tokenize(normalized_txt)
 
    #remove stop words
    #words = [w for w in words if w not in stopwords.words("english")]

    #lemmatize
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    #Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]
    
    return words


def build_model(X, Y, pipeline):
    '''
    Makes, builds, and fits a model given the X, Y, and a pipeline model.
    '''
    model = pipeline
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state = 42)
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    #make predictions
    y_preds = model.predict(X_test)

    results_dict = {}

    for pred, label, col in zip(y_preds.transpose(), Y_test.values.transpose(), Y_test.columns):
        print(col)
        print(classification_report(label, pred))
        results_dict[col] = classification_report(label, pred, output_dict=True)
    
    #return precision, recall, and f1-score of weighted averages
    weighted_avg = {}
    for key in results_dict.keys():
        weighted_avg[key] = results_dict[key]['weighted avg']

    df_wavg = pd.DataFrame(weighted_avg).transpose()
    
    print
    return df_wavg


def save_model(model_name, model_filepath):
    '''
    Saves a file to the data folder with the extension .pkl
    file path: '../data/'+ file_name+'.pkl'
    '''
    pickle.dump(file_to_pickle, open('../data/'+model_name+'.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import json
import ast

#machine learning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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
    my db filepath: 'sqlite:///../data/disaster_response.db'
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(table_name='message_categories', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_name = Y.columns.values
    return X, Y


def tokenize(text):
    '''
    Raw text tokenized via the following steps: normalized, punctuation removed, stemmed, and lemmatized
    '''
    #Normalize text and remove punctuation
    normalized_txt = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    words = word_tokenize(normalized_txt)

    #lemmatize
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    #Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]
    
    return words


def build_model(parameters=None):
    '''
    Builds and fits a pipeline model given X and Y.
    '''
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    if parameters != None:
        model_cv = GridSearchCV(estimator=model, scoring='f1_weighted', param_grid=parameters, verbose=3)
        return model_cv

    else:
        return model

def train_model(model, X_train, Y_train):
    '''
    Train model based on whether or not GridSearchCV parameters were given.
    '''
    model.fit(X_train, Y_train)
    return model



def evaluate_model(model, X_test, Y_test):
    '''
    Create a weighted averages summary dataframe for each label, 
    using the classification_report function 
    IN: 
        Y_test - array of actual values
        y_preds - numpy array of predicted values
    OUT 
        Weighted averages summary df with columns: precision, recall, f1-score, support
        Prints descriptive statistics for the f1-score, upper and lower quantile df slices
    '''
    #make predictions
    y_preds = model.predict(X_test)

    #make a dictionary of results from classificatoin_report
    results_dict = {}
    for pred, label, col in zip(y_preds.transpose(), Y_test.values.transpose(), Y_test.columns):
        results_dict[col] = classification_report(label, pred, output_dict=True)

    #extract the "weighted avg" dict of each label in the dict
    weighted_avg = {}
    for key in results_dict.keys():
        weighted_avg[key] = results_dict[key]['weighted avg']

    df_wavg = pd.DataFrame(weighted_avg).transpose() #create df from weighted avg dict
    
    print(df_wavg['f1-score'].describe()) # descriptive stats for f1-scores
    print('lowest quantile of f scores',df_wavg[df_wavg['f1-score'] <= df_wavg['f1-score'].quantile(0.25)]) # lowest quantile of f scores
    print('highest quantile of f scores', df_wavg[df_wavg['f1-score'] >= df_wavg['f1-score'].quantile(0.75)]) # highest quantile of f scores
    return df_wavg


def save_model(model_name, model_filepath):
    '''
    Saves a file to the data folder with the extension .pkl
    file path: '../data/'+ file_name+'.pkl'
    '''
    pickle.dump(model_name, open('../data/'+model_filepath+'.pkl', 'wb'))


def main(database_filepath, model_filepath, params):
    '''
    Load saved database, train a pipeline model, print evaluation metrics, and save trained model the data folder.
    '''
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model(params)

    print('Training model...')
    trained_model = train_model(model, X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(trained_model, X_test, Y_test)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(trained_model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    # Create argparser
    import argparse
    parser = argparse.ArgumentParser(description = 'Script for training classifier pipeline'\
                                     'Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier'\
              'the saved file path will be ../data/+file_name+.pkl')
    parser.add_argument("database_filepath", help = "File path for database, ../data/DisasterResponse.db")
    parser.add_argument("model_filepath", help = "File name for saving trained model, with saved file path to be ../data/+file_name+.pkl")
    parser.add_argument('-p', '--params_dict', help='Dictionary of model parameters. Dictionary should be\
                          passed in string form with values in a list, e.g. \
                          "{key: [value(s)]}". \ 
                          For windows, format as '{\"name\":key}'\
                          To see available params, use \
                          train_classifer.py database/filepath model/filepath\
                          -p', 
                            type=json.loads)
    parser.add_argument('-a', '--available_params', action='store_true', help='lists all model parameter keys')
    args = parser.parse_args()
    
    if args.available_params:
        pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
        print(pipeline.get_params().keys())
    else:
        main(database_filepath=args.database_filepath, 
             model_filepath=args.model_filepath, params=args.params_dict)
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys

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
    return X, Y, category_name


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


def build_model():
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
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
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

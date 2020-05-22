import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    IN:
        messages_filepath: csv filepath for the messages dataset
        categories_filepath: csv filepath for the categories dataset
    OUT:
        df: dataframe merged from the 2 input df's by the "id" feature
    '''
    messages = pd.read_csv(messages_filepath) #load 1st df
    categories = pd.read_csv(categories_filepath) #load 2nd df
    df = pd.merge(messages, categories, on = 'id', how = 'inner') #merge dfs
    return df

def clean_data(df):
    '''
    IN: merged dataframe of messages and categories
    OUT: cleaned dataframe with the following features:
        - category column names in clean text
        - category values encoded as boolean integers with one category per column
        - duplicates removed
    '''
    categories = df.categories.str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for col in categories:
        # set each value to be the last character of the string
        categories[col] = categories[col].str[-1]

        # convert column from string to numeric
        categories[col] = pd.to_numeric(categories[col])
        
    #drop the now unnecessary categories col
    df = df.drop('categories', axis = 1) 
    
    #concat the new dataframe to original
    df = pd.concat([df, categories], axis = 1)
    
    #drop duplicates
    df = df.drop_duplicates()
    return df
    
    
def save_data(df, database_filename):
    '''
    Save cleaned data into a SQLite db
    IN:
        df: dataframe to be saved into the SQLite database
        database_filename: string. filename for database
    OUT:
        No display
    '''
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql('message_categories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

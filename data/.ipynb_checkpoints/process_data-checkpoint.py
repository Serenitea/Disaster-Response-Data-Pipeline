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
    split_cat = df.categories.str.split(';', expand = True)
    row = split_cat.iloc[0]
    category_colnames = [x[:-2] for x in row]
    split_cat.columns = category_colnames
    for col in split_cat:
        # set each value to be the last character of the string
        split_cat[col] = split_cat[col].str[-1]

        # convert column from string to numeric
        split_cat[col] = pd.to_numeric(split_cat[col])
    
    #drop the completely empty column
    split_cat = split_cat.drop('child_alone', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_merged = pd.concat([df, split_cat], axis = 1)

    # drop the original categories column from `df`
    df_merged = df_merged.drop(['categories'], axis = 1)

    #drop duplicates
    df_merged = df_merged.drop_duplicates()
    
    # Remove rows with a related value of 2 from the dataset
    df_merged = df_merged[df_merged['related'] != 2]
    
    return df_merged

    
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
    df.to_sql('message_categories', engine, index=False, if_exists='replace')


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

import json
import plotly
import pandas as pd
import re
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
from plotly.express import bar
from plotly.basedatatypes import BaseFigure
from sqlalchemy import create_engine

#create app
app = Flask(__name__)

#create functions
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
    clean_tokens = [PorterStemmer().stem(w) for w in words]
    
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table(table_name='message_categories', con=engine)

# load model
model = joblib.load("../models/model_3.pkl")

#colors = ["#4CB391", "azure"]
#colors = px.colors.sequential.Plasma
#colors = ['#a3a7e4'] * 100
colors = px.colors.cyclical.IceFire
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    global colors
    # extract data needed for visuals
    data = df.iloc[:, 4:]
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_count = (df.iloc[:, 4:] != 0).sum()
    cat_names = list(cat_count.index)
    
    
    graphs = [
        {
            'data': [
                bar(
                    x=genre_names,
                    y=genre_counts,
                    color=genre_counts
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
    #top 5 most popular categories
        {
            'data': [
                bar(
                    x=cat_names,
                    y=cat_count,
                    color=cat_count, color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2
                )
            ],

            'layout': {
                'title': 'Counts of All Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }

        
    ]

    
    #corr_list = []
    #for row in data.corr().values:
    #    corr_list.append(list(row))
    
    # create visuals
    #strength
    
    #Default graph: genre distribution graph
    '''
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker = dict(color=color)
                    
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
    #top 5 most popular categories
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_count,
                    marker = dict(color=color)
                )
            ],

            'layout': {
                'title': 'Counts of All Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }

        
    ]
    '''

    #assign unique id to each graph
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]

    graphJSON = BaseFigure.write_json('graph-1')
    # encode plotly graphs in JSON
    #graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()

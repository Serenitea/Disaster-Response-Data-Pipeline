
### Table of Contents

1. [Run Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Project Pipeline](#pipeline)
5. [Results](#Results)
6. [Licensing, Authors, and Acknowledgements](#licensing)# Disaster Response Pipeline Project

##Run Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

## Project Description<a name="ProjectDescription"></a>

Figure Eight Data Set provides a set of prelabelled [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/). The data contains 30,000 messages that are drawn from natural disaster events such as earthquakes, floods, and storms around the globe. have been encoded with 36 different categories, such as "medical_help", "refugees", "fire", "shelter", etc.

The aim of this project is to create an app that can help emergency workers analyze messages to help with efficient distribution of emergency resources.

## File Description <a name="FileDescription"></a>

4 main folders:
1. app
- run.py: Launches and loads data into the Flask webapp.
- templates: html templates for the webapp.
2. data
- original dataset: disaster_categories.csv, disaster_messages.csv
- process_data.py: ETL pipeline scripts to read, clean, and save data into an SQL database
- disaster_response.db: SQL database  of cleaned messages and categories data, output of the ETL pipeline
3. models
- train_classifier.py: machine learning pipeline scripts to train and export a classifier
- model_*.pkl: pickled file of the trained machine learning classifier
4. notebooks
- ETL Pipeline Preparation.ipynb: detailed Jupyter notebook of my ETL pipeline work.
- ML Pipeline Preparation.ipynb: detailed Jupyter notebook of my ML pipeline work.

## Project Pipeline <a name="pipeline"></a>

1. Data Preprocessing
2. Building Pipeline classifier model
- transforms: countevectorizer with custom tokenize function, tfidtransformer
- estimator: AdaBoostClassifier
3. Evaluating model - using sklearn's classification report to compare F1 scores, precision, and recall
4. Improving model - GridSearchCV
5. Exporting model as .pkl file

## Results <a name="Results"></a>
1. Created an ETL pipeline to load and merge data from 2 csv files, clean data, and save data into a SQLite database.
2. Created a machine learning pipeline to train a multi-output classifier that categorizes text
3. Created a Flask app that can classify any message that users enter on the web page.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credits to Figure Eight for the [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) data set. This project is part of a Data Science nanodegree program with Udacity.

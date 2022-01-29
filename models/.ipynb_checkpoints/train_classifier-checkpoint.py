import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

# import statements
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """ Loads sql database table from ETL
    
    Args:
    database_filepath: sqlite table 
    
    Returns:
    X: features text
    Y: all dependent variables of outcomes
    category_names: list of the Y categories
    
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(f'{database_filepath}', engine)
    df = df[df.related != 2]
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names
    
    


def tokenize(text):
    """ A function that clean the messages string using NLP 
    
    Args:
    text: an array of texts
    
    Returns: 
    
    text: a clean, preprocessed version of orginal text
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'   
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)  
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    text = clean_tokens
    #clean_tokens = [x for x in clean_tokens if x not in punc]    
    return text

def build_model():
    """  ML Function Pipline 
    This function has sklearn Pipline that take two transformer and an multioutput classifier estimator.
    Also, a paramater to optimize and find best model tunining for better accuracy
    
    Returns:
    model: a fited model that has been computed with Cross-Validation Search of all the paramaters
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    parameters = {
        "clf__estimator__n_estimators": [100],
        "clf__estimator__max_depth": [None],
        "clf__estimator__max_features": ["sqrt"]
    }
    
    cv = GridSearchCV(pipeline, parameters)
    model = cv
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate the ML model.
    Arguments: 
    model: ready to test model
    X_test: feature to test on the model
    Y_test: dependent outputs to test the model
    category_names
    Returns: 
    None
    """  
    
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.columns = Y_test.columns
    Y_pred.index = Y_test.index
    
    print(classification_report(Y_test, Y_pred, target_names=Y_pred.columns))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """ Function to save the pickle result of the model
    Args:
    model: ML model
    model_filepath: a path to save the pickle file
    
    """
    pickle.dump(model,open(model_filepath,'wb')) 
    


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
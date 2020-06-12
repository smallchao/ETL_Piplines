from sqlalchemy import create_engine
import re
import pickle
import nltk

import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer,confusion_matrix,classification_report,fbeta_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    '''
    加载数据库文件
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterData', engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns.values
    return X, y, category_names

def tokenize(text):
    '''
    文本分词
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    tokens = [WordNetLemmatizer().lemmatize(word) for word in words if word not in stop_words]

    return tokens
 
def build_model_0():
    '''
    基础ML管道+网格搜索
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('transformer', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(DecisionTreeClassifier(random_state=10),n_jobs=-1))
    ])

    # create grid search object
    parameters = {
        'clf__estimator__min_samples_split':[3,4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv

def build_model():
    '''
    改进的ML管道+网格搜索
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('transformer', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(RandomForestClassifier(random_state=10),n_jobs=-1))
    ])

    # create grid search object
    parameters = {
        'clf__estimator__min_samples_split':[3,4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    模型验证
    '''
    y_pred = model.predict(X_test)
    for i in range(10):
        print(classification_report(y_test[:,i],y_pred[:,i]))
    for i in range(10):
        print(accuracy_score(y_test[:,i],y_pred[:,i]))
 

def save_model(model, model_filepath):
    '''
    模型保存
    '''
    pickle.dump(pipeline, open('model.pkl', 'wb'))


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

   


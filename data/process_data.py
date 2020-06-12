import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    加载数据集并进行数据组装
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = ['id'])

    return df

def clean_data(df):
    '''
    数据清洗
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]

    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    df.drop_duplicates(inplace = True)
   
    return df

def save_data(df, database_filename):
    '''
    将数据保存为数据库文件
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')

def main():

    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning...')
        df = clean_data(df)
        print('Saving...in {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('finish!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterData.db')

if __name__ == '__main__':

    main()
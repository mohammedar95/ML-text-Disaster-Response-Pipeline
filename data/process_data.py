import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """ Function to load the original source data filepath
    Args:
    messages_filepath: raw data of text messages file path
    categories_filepath: data source of categories path

    Returns:
    df = a merged dataframe from messages and categories
    """
    # Load messages.csv into a dataframe
    # Load categories.csv into a dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """ Function to clean and pre-process the text data and transform it
    Args:
    df: merged dataframe from

    Returns:
    df: post-processed with cleaned and dropped duplicated
    """

    # Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df.categories.str.split(';', expand=True)

    #Use the first row of categories dataframe to create column names for the categories data.
    row = categories.head(1).values.tolist()

    lst = []
    for i in row:
        for j in i:
            lst.append(j[:-2])

    category_colnames = lst

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:
        categories[column] = [int(x[-1:]) for x in categories[column]]


    categories.related = categories.related.replace(2,1)
    # drop the original categories column from `df`
    df = df.drop("categories", axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)

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

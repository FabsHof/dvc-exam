import logging
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def split_data(raw_data_path="./data/raw/raw.csv", processed_data_path="./data/processed"):
    ''' Split raw data into train and test sets and save them in the appropriate folders.
    Produces the following 4 files: X_test.csv, X_train.csv, y_test.csv, y_train.csv.'''
    # Load raw data
    df = pd.read_csv(raw_data_path)

    # Target and feature separation
    target = df['silica_concentrate']
    features = df.drop(columns=['silica_concentrate'], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Create data/processed directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Save the datasets
    X_train.to_csv(os.path.join(processed_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)
    
    logging.info("Data has been split into train and test sets.")

def main():
    """ Split data into train and test sets
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting data into train and test sets')
    split_data()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
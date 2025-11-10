import logging
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(processed_data_path="./data/processed"):
    ''' Normalize feature data using StandardScaler and save the normalized datasets.
    Produces the following 2 files: X_test_scaled.csv, X_train_scaled.csv.'''
    # Load feature datasets
    X_train_path = os.path.join(processed_data_path, 'X_train.csv')
    X_test_path = os.path.join(processed_data_path, 'X_test.csv')
    
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both training and test data
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test_normalized_df = pd.DataFrame(X_test_normalized, columns=X_test.columns)
    
    # Save the normalized datasets
    X_train_normalized_df.to_csv(os.path.join(processed_data_path, 'X_train_scaled.csv'), index=False)
    X_test_normalized_df.to_csv(os.path.join(processed_data_path, 'X_test_scaled.csv'), index=False)
    
    logging.info("Feature data has been scaled.")

def main():
    logger = logging.getLogger(__name__)
    logger.info('Scaling feature data')
    normalize_data()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
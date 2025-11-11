import logging
import os
import pandas as pd
from sklearn.linear_model import Ridge

def do_model_training(model=Ridge(), processed_data_path="./data/processed_data", model_path="./models"):
    ''' Train the model with the best hyperparameters on the training data and save the trained model.'''
    # Load training data
    X_train_path = os.path.join(processed_data_path, 'X_train_scaled.csv')
    y_train_path = os.path.join(processed_data_path, 'y_train.csv')
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # Load best hyperparameters and instantiate the model
    best_params_path = os.path.join(model_path, 'best_params.pkl')
    best_params = pd.read_pickle(best_params_path)
    logging.info(f'Loading best hyperparameters: {best_params}')
    model.set_params(**best_params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model
    trained_model_path = os.path.join(model_path, 'trained_model.pkl')
    pd.to_pickle(model, trained_model_path)
    logging.info(f'Model trained and saved at {trained_model_path}')

def main():
    """ Train the model with the best hyperparameters on the training data
    """
    logger = logging.getLogger(__name__)
    logger.info('Training the model with the best hyperparameters')
    
    do_model_training()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
import logging
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


def grid_search(model=Ridge(), search_params={
    'alpha': [0.1, 1, 10, 100],
}, processed_data_path="./data/processed_data", model_path="./models"):
    ''' Perform a grid search for Ridge Regression model and save the best parameters.'''
    grid_search = GridSearchCV(estimator=model, param_grid=search_params, cv=5)

    X_train_path = os.path.join(processed_data_path, 'X_train_scaled.csv')
    y_train_path = os.path.join(processed_data_path, 'y_train.csv')
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Save best parameters to a pkl file
    best_params_path = os.path.join(model_path, 'best_params.pkl')
    pd.to_pickle(best_params, best_params_path)
    logging.info(f"Best parameters found: {best_params}")

def main():
    """ Define grid search parameters for model tuning
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Defining grid search parameters')
    
    grid_search()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
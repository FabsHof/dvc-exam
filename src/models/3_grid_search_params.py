import logging
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


def do_grid_search(model=None, grid_search_params={
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}, processed_data_path="./data/processed", model_path="./models"):
    ''' Perform grid search to find the best hyperparameters for a model.'''
    if model is None:
        model = LinearRegression()
    
    X_train_path = os.path.join(processed_data_path, 'X_train_scaled.csv')
    y_train_path = os.path.join(processed_data_path, 'y_train.csv')
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    grid_search = GridSearchCV(estimator=model, param_grid=grid_search_params, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Save best parameters to a pkl file
    best_params_path = os.path.join(model_path, 'best_params.pkl')
    pd.to_pickle(best_params, best_params_path)
    logging.info(f"Best parameters found: {best_params}")





def main(model=None, processed_data_path="./data/processed"):
    """ Define grid search parameters for model tuning
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Defining grid search parameters')
    
    logger.info(f'Start grid search with parameters: {grid_search_params}')

    # Example grid search parameters
    grid_search_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    do_grid_search(model, grid_search_params, processed_data_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
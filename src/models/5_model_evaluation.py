import logging
import json
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(processed_data_path="./data/processed_data", model_path="./models", metrics_path="./metrics"):
    ''' Evaluate the trained model on the test dataset and log the performance metrics.'''

    # Load test data
    X_test_path = os.path.join(processed_data_path, 'X_test_scaled.csv')
    y_test_path = os.path.join(processed_data_path, 'y_test.csv')
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    # Load the trained model
    trained_model_path = os.path.join(model_path, 'trained_model.pkl')
    model = pd.read_pickle(trained_model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Store predictions
    predictions_path = os.path.join(metrics_path, 'predictions.csv')
    os.makedirs(metrics_path, exist_ok=True)
    metrics_file = pd.concat([X_test, pd.DataFrame({'actual': y_test.squeeze(), 'predicted': y_pred})], axis=1)
    metrics_file.to_csv(predictions_path, index=False)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics to a file
    os.makedirs(metrics_path, exist_ok=True)
    metrics_file_path = os.path.join(metrics_path, 'scores.json')
    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f)
    
    # Log the evaluation metrics
    logging.info(f'Model Evaluation Metrics:')
    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'R^2 Score: {r2}')

def main():
    logger = logging.getLogger(__name__)
    logger.info('Evaluating the trained model')
    evaluate_model()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
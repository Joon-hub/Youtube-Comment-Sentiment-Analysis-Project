import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str):
    """Load the vectorized data."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
            # Add a type check to ensure we loaded a vectorizer
            if not isinstance(vectorizer, TfidfVectorizer):
                raise TypeError(f"Expected TfidfVectorizer but got {type(vectorizer)}")
        logger.debug('Vectorized data loaded successfully')
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorized data: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def main():
    """Main function to orchestrate model evaluation pipeline."""
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # model and vectorize X_test using tfidf
        vectorizer = load_vectorizer(os.path.join(root_dir, 'models/tfidf_vectorizer.pkl'))
        model = load_model(os.path.join(root_dir, 'models/xgb_model.pkl'))

        # prepare the data
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
        # transform the data to tfidf
        X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values

        # Evaluate model and get metrics
        clf_report, cm = evaluate_model(model, X_test_tfidf, y_test)

        logger.debug('Confusion matrix: %s', clf_report)
        logger.debug('Classification report: %s', cm)

        print("\n Classification report:\n")
        print(classification_report(y_test, model.predict(X_test_tfidf)))

        print("\nConfusion matrix:\n")
        print(cm)

        logger.debug('Model evaluation completed successfully')
    except Exception as e:
        logger.error('Error in main execution: %s', e)
        raise

if __name__ == "__main__":
    main()

    
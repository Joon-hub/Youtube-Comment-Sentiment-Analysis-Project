import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
import sys

# Enhanced logging setup with debug information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_building_debug.log', mode='w')
    ]
)
logger = logging.getLogger('model_building')

def get_project_paths():
    """Get all relevant project paths."""
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = {
            'root': root_dir,
            'models': os.path.join(root_dir, 'models'),
            'params': os.path.join(root_dir, 'params.yaml'),
            'train_data': os.path.join(root_dir, 'data/interim/train_processed.csv')
        }
        logger.debug('Project paths initialized: %s', paths)
        return paths
    except Exception as e:
        logger.error('Failed to initialize project paths: %s', e, exc_info=True)
        raise

def ensure_directories_exist(paths):
    """Ensure all required directories exist."""
    try:
        os.makedirs(paths['models'], exist_ok=True)
        logger.debug('Models directory confirmed at: %s', paths['models'])
        
        # Verify directory permissions
        if not os.access(paths['models'], os.W_OK):
            raise PermissionError(f"No write permission for models directory: {paths['models']}")
    except Exception as e:
        logger.error('Directory creation/verification failed: %s', e, exc_info=True)
        raise

def load_params(params_path):
    """Load parameters from YAML file."""
    try:
        logger.debug('Attempting to load parameters from: %s', params_path)
        with open(params_path) as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters loaded successfully: %s', params)
            return params
    except Exception as e:
        logger.error('Failed to load parameters: %s', e, exc_info=True)
        raise

def load_data(file_path):
    """Load and prepare training data."""
    try:
        logger.debug('Loading data from: %s', file_path)
        df = pd.read_csv(file_path)
        logger.debug('Data shape before NaN handling: %s', df.shape)
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        logger.debug('NaN counts before filling: %s', nan_counts)
        
        df = df.fillna('')
        
        logger.debug('Data loaded successfully. Shape: %s', df.shape)
        logger.debug('Column names: %s', df.columns.tolist())
        logger.debug('Data types: %s', df.dtypes)
        
        # Verify essential columns
        required_columns = ['clean_comment', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
    except Exception as e:
        logger.error('Data loading failed: %s', e, exc_info=True)
        raise

def apply_tfidf(train_data, params):
    """Apply TF-IDF transformation to training data."""
    try:
        logger.debug('Initializing TF-IDF with params: %s', 
                    {'max_features': params['max_features'], 
                     'ngram_range': params['ngram_range']})
        
        # Initialize vectorizer with params:
        vectorizer = TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=tuple(params['ngram_range'])
        )

        # Apply TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(train_data['clean_comment'].values)
        y_train = train_data['category'].values
        logger.debug('TF-IDF transformation completed. Shape: %s', X_train_tfidf.shape)

        # Debug transformation results
        logger.debug('Vocabulary size: %d', len(vectorizer.vocabulary_))
        logger.debug('Features shape: %s', X_train_tfidf.shape)
        logger.debug('Labels shape: %s', y_train.shape)
        logger.debug('Unique labels: %s', np.unique(y_train))

        # Save vectorizer for future use
        vectorizer_path = os.path.join(paths['models'],'tfidf_vectorizer.pkl')
        logger.debug('Saving tf-idf vectorizer to %s' % vectorizer_path)
        with open(vectorizer_path,'wb') as f:
            pickle.dump(vectorizer, f)
        # Verify file was created
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not created at: {vectorizer_path}")
        
        logger.debug('Vectorizer saved successfully')
        
        # Return transformed features and labels
        return X_train_tfidf, y_train
        
    except Exception as e:
        logger.error('TF-IDF transformation failed: %s', e, exc_info=True)
        raise
def apply_adasyn(X_train, y_train):
    """
    Apply ADASYN oversampling to handle class imbalance.
    """
    try:
        logger.debug('Applying ADASYN to balance dataset.')
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
             
        # Debug information about resampled data
        logger.debug('Original dataset shape: %s', X_train.shape)
        logger.debug('Resampled dataset shape: %s', X_resampled.shape)
        logger.debug('Original label distribution: %s', dict(zip(*np.unique(y_train, return_counts=True))))
        logger.debug('Resampled label distribution: %s', dict(zip(*np.unique(y_resampled, return_counts=True))))
        
        return X_resampled, y_resampled
    except Exception as e:
        logger.error('ADASYN application failed: %s', e, exc_info=True)
        raise

def train_model(X_train, y_train, params):
    """Train and save XGBoost model."""
    try:
        logger.debug('Initializing XGBoost with params: %s', params)
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=2,
            subsample=0.7993530839476384,
            colsample_bytree=0.6762670388056441,
            gamma=0.5119258999221035,
            reg_alpha=0.4307342241387505,
            reg_lambda=0.4444037679100022,
            random_state=41
        )
        logger.debug('Starting model training')
        logger.debug('Training data shape: %s', X_train.shape)
        logger.debug('Labels shape: %s', y_train.shape)
        
        model.fit(X_train, y_train)
        
        # Debug model information
        logger.debug('Model training completed')
        logger.debug('Feature importances shape: %s', model.feature_importances_.shape)
        
        # Save model
        model_path = os.path.join(paths['models'], 'xgb_model.pkl')
        logger.debug('Saving model to: %s', model_path)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Verify file was created
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not created at: {model_path}")
            
        logger.debug('Model saved successfully')
        return model
        
    except Exception as e:
        logger.error('Model training failed: %s', e, exc_info=True)
        raise

def main():
    try:
        logger.info('Starting model building pipeline')
        
        # Ensure models directory exists
        logger.debug('Checking directories')
        ensure_directories_exist(paths)
        
        # Load parameters and data
        logger.debug('Loading parameters and data')
        params = load_params(paths['params'])['model_building']
        train_data = load_data(paths['train_data'])
        
        # Process TF-IDF Transformation
        logger.debug('Starting TF-IDF transformation')
        X_train, y_train = apply_tfidf(train_data, params)

        # Apply ADASYN for class balancing
        logger.debug('Starting ADASYN for class balancing')
        X_train, y_train = apply_adasyn(X_train, y_train)
        
        # Training the mode
        logger.debug('Starting model training')
        train_model(X_train, y_train, params)
        
        logger.info('Pipeline completed successfully')
        
    except Exception as e:
        logger.error('Pipeline failed: %s', e, exc_info=True)
        raise
    finally:
        logger.debug('Pipeline execution finished')

if __name__ == '__main__':
    try:
        paths = get_project_paths()
        main()
    except Exception as e:
        logger.critical('Fatal error: %s', e, exc_info=True)
        sys.exit(1)
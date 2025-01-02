import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna('')
    return df

def apply_tfidf(train_data, params):
    vectorizer = TfidfVectorizer(
        max_features=params['max_features'],
        ngram_range=tuple(params['ngram_range'])
    )
    X_train_tfidf = vectorizer.fit_transform(train_data['clean_comment'].values)
    y_train = train_data['category'].values
    return X_train_tfidf, y_train, vectorizer

def apply_adasyn(X_train, y_train):
    adasyn = ADASYN(random_state=42)
    return adasyn.fit_resample(X_train, y_train)

def objective(trial, X_train, y_train):
    # Hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0)
    }

    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)
        score = accuracy_score(y_val_cv, y_pred)
        scores.append(score)

    return np.mean(scores)

def train_with_optuna(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    best_params = study.best_params
    model = xgb.XGBClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)
    return model

def main():
    try:
        logger.info('Starting model building pipeline')
        
        with open('params.yaml') as f:
            params = yaml.safe_load(f)['model_building']
        
        train_data = load_data('data/interim/train_processed.csv')
        X_train, y_train, vectorizer = apply_tfidf(train_data, params)
        X_train, y_train = apply_adasyn(X_train, y_train)
        model = train_with_optuna(X_train, y_train)
        
        os.makedirs('models', exist_ok=True)
        with open('models/xgb_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        logger.info('Pipeline completed successfully')
    except Exception as e:
        logger.error(f'Pipeline failed: {str(e)}')
        raise

if __name__ == '__main__':
    main()

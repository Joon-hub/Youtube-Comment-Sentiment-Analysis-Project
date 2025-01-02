from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This capstone project focuses on building an end-to-end machine learning pipeline for automated comment analysis, leveraging modern MLOps practices to ensure scalability, reproducibility, and deployment readiness. The project involves preprocessing user comments using NLP techniques like TF-IDF vectorization, addressing class imbalances with ADASYN, and optimizing model performance through hyperparameter tuning with Optuna. It employs machine learning algorithms such as XGBoost, with experiments, metrics, and models tracked using MLflow for efficient workflow management. The solution integrates containerized environments using Docker and is deployed on AWS EC2 instances, with S3 buckets utilized for secure and scalable storage of data and model artifacts. This ensures a robust, scalable system for classifying and analyzing user comments with high accuracy, suitable for real-world applications.',
    author='sudhir joon',
    license='',
)

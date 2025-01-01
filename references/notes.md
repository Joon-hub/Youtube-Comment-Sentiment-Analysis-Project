# Notes

## Create a Cookiecutter Template
To create a data science project structure using cookiecutter:

1. Run the following command:
   ```bash
   cookiecutter https://github.com/drivendata/cookiecutter-data-science
2. create a virtual environment and activate it 
3. create a github repository
4. git init
5. git remote add origin <repository-url>
6. git push origin main -f

# DVC Pipeline

## Stage 1: Data Ingestion
- Fetch data from the specified URL.
- Perform basic data cleaning:
  - Drop missing values.
  - Drop duplicates.
  - Remove empty strings.
  - Perform train-test split.
- Store the cleaned data in the `data/raw` folder.

## Stage 2: Preprocessing
- Fetch train and test data from the `data/raw` folder.
- Perform preprocessing:
  - Convert text to lowercase.
  - Remove URLs.
  - Remove stop words.
  - Perform lemmatization.
- Store the preprocessed data in the `data/interim` folder.

## Stage 3: Model Building
- Fetch train and test data from the `data/interim` folder.
- Apply TF-IDF vectorization with:
  - Bigrams.
  - Maximum features set to 1000.
- Build the model using XGBoost with optimal parameters.
- Save the following:
  - Model: `model.pkl`
  - Vectorizer: `vectorizer.pkl`

## Stage 4: Model Evaluation
- Fetch the test data from the `data/interim` folder.
- Load the trained model and make predictions on the test data.
- Print the classification report.

---

## Configuration Files

### `dvc.yaml`
- Used to connect all the pipeline stages.

### `params.yaml`
- Contains the parameters for the pipeline.

---

## Variables
- **Train-Test Split:** Ratio for splitting the data.
- **N-grams:** Unigram, Bigram, Trigram.
- **Maximum Features:** Limit for TF-IDF vectorizer.
- **XGBoost Parameters:** Hyperparameters for model optimization.

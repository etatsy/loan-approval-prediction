# Loan Approval Prediction

This project is a machine learning practice project for predicting loan approval status based on applicant and loan-related information.

The main goal of the project is to practice a complete ML workflow: loading data, exploring missing values and duplicates, cleaning the dataset, building preprocessing pipelines, training a classification model, and evaluating model performance.

## Project Description

Loan approval prediction is a binary classification task where the model predicts whether a loan application will be approved or rejected.

The project focuses not only on model training, but also on the full preparation process before modeling: checking missing values, analyzing duplicates, deciding how to handle outliers, splitting the data correctly, and building preprocessing pipelines that help avoid data leakage.

## Project Structure

```text
loan-approval-prediction/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── loans_modified.csv
│   └── processed/
├── models/
│   └── champion_model_metadata.json
├── notebooks/
│   └── 01_loan_approval_workbook.ipynb
├── reports/
├── scripts/
│   ├── start_mlflow_ui.sh
│   └── start_streamlit_app.sh
├── requirements.txt
└── README.md
```

## Workflow

The notebook follows the main steps of a typical supervised machine learning project:

1. Load the loan approval dataset
2. Explore dataset structure, missing values, and duplicates
3. Remove fully duplicated rows
4. Analyze duplicated `loan_id` values
5. Remove the `loan_id` column before modeling
6. Remove rows with missing target values
7. Analyze numerical outliers
8. Keep outliers when they are realistic business values
9. Split the data into training and test sets with stratification
10. Build preprocessing pipelines for numerical and categorical features
11. Train a classification model using a full sklearn pipeline
12. Evaluate the model with classification metrics
13. Use cross-validation to check model stability

## Data Cleaning

The dataset contains missing values, duplicated rows, and an identifier column.

The cleaning process includes:

- removing fully duplicated rows
- checking duplicated `loan_id` values
- keeping rows where `loan_id` is missing because the identifier is not used as a model feature
- removing the `loan_id` column before modeling
- removing rows where the target variable `loan_status` is missing

The target column is not imputed because it represents the value the model is supposed to predict.

## Outlier Analysis

Numerical columns were checked for outliers using the IQR method and boxplots.

The detected outliers were not removed automatically because values such as applicant income, coapplicant income, loan amount, and loan term can be realistic business values.

The project treats outliers as potentially useful signals rather than errors unless the values are clearly impossible.

## Preprocessing

Numerical features are processed with:

- `SimpleImputer(strategy="median")`
- `MinMaxScaler`

Categorical features are processed with:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

The preprocessing steps are combined using `ColumnTransformer`.

The full preprocessing workflow is included inside an sklearn `Pipeline` to make sure the same transformations are applied to both training and test data.

## Model

The baseline model is:

- `RandomForestClassifier`

The model is trained using a full pipeline:

1. preprocessing numerical and categorical columns
2. fitting the classifier
3. predicting loan approval status

## Cross-Validation

Cross-validation is used to check how stable the model performance is across different splits of the training data.

The project uses stratified cross-validation because the target classes are imbalanced.

The final test set is kept separate and used only for the final evaluation.

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Cross-validation metrics

Because the target classes are imbalanced, extra attention is paid to the minority class performance.

## Key Learning Points

This project was built mainly to practice:

- loading data with pandas
- exploratory data analysis
- handling missing values
- duplicate analysis
- outlier analysis
- train/test split
- stratified splitting
- preventing data leakage
- building sklearn pipelines
- using `ColumnTransformer`
- encoding categorical variables
- imputing missing values
- applying cross-validation
- interpreting classification metrics

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Repository Description

Loan approval prediction with data cleaning, preprocessing pipelines, cross-validation, and model evaluation.

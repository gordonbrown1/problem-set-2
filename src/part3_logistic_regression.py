'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''



# Import any further packages you may need for PART 3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr
import warnings
import sys
import io

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Create context manager to suppress stderr (Cython warnings)
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    """
    Context manager to temporarily suppress stderr output.
    This function redirects stderr to a null stream, effectively hiding
    all warning messages that would normally be printed to the console.
    
    Yields:
        None: The context manager yields control back to the calling code
              while stderr is suppressed.
    
    Notes:
        Useful for hiding Cython-generated warnings
        that cannot be suppressed by warnings.filterwarnings()
    """
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


# Your code here
def main():
    """
    Main function for PART 3: Logistic Regression.
    
    This function executes the complete logistic regression pipeline:
        1. Loads preprocessed data from ./data/df_arrests.csv
        2. Splits data into train (70%) and test (30%) sets with stratification
        3. Performs hyperparameter tuning for C using GridSearchCV with 5-fold CV
        4. Identifies optimal C value and regularization level
        5. Makes predictions on test set
        6. Saves predictions to ./data/lr_predictions.csv
        7. Calculates and prints test ROC-AUC and PPV for top 50 highest risk
        8. Returns train, test, and results dataframes for use in PART 4 and 5
    
    Returns:
        tuple: (df_arrests_train, df_arrests_test, df_results) where:
            df_arrests_train (DataFrame): Training set with all columns
            df_arrests_test (DataFrame): Test set with all columns  
            df_results (DataFrame): Results with person_id, features, true labels, and pred_lr
    
    """
    print("\n")
    print("="*50)
    print("LOGISTIC REGRESSION")
    print("="*50)
    
    # Read in df_arrests
    print("\nLoading data (df_arrests)...")
    df_arrests = pd.read_csv('./data/df_arrests.csv')
    print(f"df_arrests shape: {df_arrests.shape}")
    
    # Create features list (pred_universe is current_charge_felony)
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    #print(f"\nFeatures: {features}")
    
    # Prepare X and y
    X = df_arrests[features]
    y = df_arrests['y']
    
    # Create train/test split (test_size=0.3, shuffle=True, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=0.3,random_state=42,shuffle=True,stratify=y)
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training data class/label distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing data class/label distribution: {y_test.value_counts().to_dict()}\n")
    
    # Create parameter grid with three values for C
    param_grid = {'C': [0.1, 1, 10]}
    print(f"\nChosen parameter grid set: {param_grid}")
    
    # Initialize Logistic Regression model
    lr_model = lr(random_state=42)
    
    # Initialize GridSearchCV with 5-fold cross-validation
    gs_cv = GridSearchCV(
        estimator=lr_model,param_grid=param_grid,
        cv=5,scoring='roc_auc',n_jobs=1,verbose=0)
    
    # Run the model with suppressed warnings
    print("\nRunning GridSearchCV with 5-fold cross-validation...")
    with suppress_stderr():
        gs_cv.fit(X_train, y_train)
    
    # Get best parameters
    best_C = gs_cv.best_params_['C']
    best_score = gs_cv.best_score_
    
    print(f"\nAverage score for best hyper-parameter (ROC-AUC): {best_score:.4f}")
    
    # Answer questions about C
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    print(f"\nQuestion: What was the optimal value for C?")
    print(f"    Answer: {best_C}")
    
    # Determine regularization level
    if best_C == min(param_grid['C']):
        regularization = "MOST regularization (smallest C value)"
    elif best_C == max(param_grid['C']):
        regularization = "LEAST regularization (largest C value)"
    else:
        regularization = "MIDDLE regularization"
    
    print(f"\nQuestion: Did it have the most or least regularization? Or in the middle?")
    print(f"    Answer: {regularization}")
    
    # Predict for the test set with suppressed warnings
    
    with suppress_stderr():
        y_pred_proba = gs_cv.predict_proba(X_test)[:, 1]
    
    # Create results dataframe with ONLY what's needed for Parts 4 and 5
    df_results = pd.DataFrame({
        'person_id': df_arrests.loc[X_test.index, 'person_id'].values,
        'current_charge_felony': X_test['current_charge_felony'].values,
        'num_fel_arrests_last_year': X_test['num_fel_arrests_last_year'].values,
        'y': y_test.values, 'pred_lr': y_pred_proba,})
    
    # Save ONLY the results needed for Parts 4 and 5
    df_results.to_csv('./data/lr_predictions.csv', index=False)
    print(f"\nSaved predictions to ./data/lr_predictions.csv")
    print(f"  Shape: {df_results.shape}")
    print(f"  Columns: {list(df_results.columns)}")
    
    # Calculate and print AUC
    auc_score = gs_cv.score(X_test, y_test)
    print(f"\nTest ROC-AUC: {auc_score:.4f}")
    
    # Calculate and print PPV for top 50
    top_50 = df_results.nlargest(50, 'pred_lr')
    ppv = top_50['y'].mean()
    print(f"PPV for Top 50 Highest Risk: {ppv:.4f} ({top_50['y'].sum()}/{len(top_50)})")
    
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION COMPLETE")
    print("="*50)
    
    # Return dataframes for use in main.py
    df_arrests_train = df_arrests.loc[X_train.index].copy()
    df_arrests_test = df_arrests.loc[X_test.index].copy()
    
    return df_arrests_train, df_arrests_test, df_results


if __name__ == "__main__":
    main()
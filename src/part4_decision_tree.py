'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
import warnings
import sys
import io

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Create context manager to suppress stderr
from contextlib import contextmanager


"""
PART 4: Decision Trees
- Train a decision tree model to predict felony rearrest
- Use GridSearchCV to tune the max_depth hyperparameter
- Evaluate on test set and save predictions


# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
import warnings
import sys
import io

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Create context manager to suppress stderr
from contextlib import contextmanager

"""

@contextmanager
def suppress_stderr():
    """
    Context manager that temporarily suppresses stderr output.
    
    This function redirects stderr to a null stream during model training
    to hide harmless warnings. After the context block exits, stderr is restored.
    
    Yields:
        None: Control back to the calling code while stderr is suppressed.
    """
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


def load_data():
    """
    Load the preprocessed data from part2_preprocessing.
    
    Returns:
        DataFrame: df_arrests with all features and target
    """
    print("\nLoading data (df_arrests)...")
    df_arrests = pd.read_csv('./data/df_arrests.csv')
    #print(f"df_arrests shape: {df_arrests.shape}")
    
    return df_arrests


def create_train_test_split(df_arrests):
    """
    Create train/test split with stratification by the outcome.
    
    Args:
        df_arrests: Full dataframe with target variable
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df_arrests_train, df_arrests_test)
    """
    print("\n" + "="*50)
    print("CREATING TRAIN/TEST SPLIT")
    print("="*50)
    
    # Define features (same as logistic regression)
    feature_cols = ['current_charge_felony', 'num_fel_arrests_last_year']
    
    X = df_arrests[feature_cols]
    y = df_arrests['y']
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: y")
    print(f"Total samples: {len(df_arrests)}")
    print(f"Class distribution:")
    print(f"  y=0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  y=1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    # Create train/test split with stratification (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=0.3,random_state=42,shuffle=True,stratify=y)
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training data class/label distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing data class/label distribution: {y_test.value_counts().to_dict()}\n")
    
    # Create the full train and test dataframes for reference
    df_arrests_train = df_arrests.loc[X_train.index].copy()
    df_arrests_test = df_arrests.loc[X_test.index].copy()
    
    return X_train, X_test, y_train, y_test, df_arrests_train, df_arrests_test


def train_decision_tree(X_train, y_train):
    """
    Train decision tree model with GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        tuple: (gs_cv_dt, best_max_depth, best_score)
    """
    print("\n" + "="*50)
    print("TRAINING DECISION TREE MODEL")
    print("="*50)
    
    # Create parameter grid with three values for max_depth
    # max_depth controls tree complexity (smaller = more regularization)
    param_grid_dt = {'max_depth': [3, 5, 7]}
    
    print(f"\nParameter grid:")
    print(f"  max_depth values: {param_grid_dt['max_depth']}")
    
    # Initialize Decision Tree model
    dt_model = DTC(random_state=42, class_weight='balanced')
    
    # Initialize GridSearchCV with 5-fold cross-validation
    gs_cv_dt = GridSearchCV(
        estimator=dt_model, param_grid=param_grid_dt, cv=5,
        scoring='roc_auc',n_jobs=1,verbose=0)
    
    # Run the model with suppressed warnings
    print("\nRunning GridSearchCV with 5-fold cross-validation...")
    with suppress_stderr():
        gs_cv_dt.fit(X_train, y_train)
    
    # Get best parameters
    best_max_depth = gs_cv_dt.best_params_['max_depth']
    best_score = gs_cv_dt.best_score_
    
    print(f"\nAverage score for best hyper-parameter (ROC-AUC): {best_score:.4f}")
    
    return gs_cv_dt, best_max_depth, best_score


def analyze_hyperparameter_results(best_max_depth, param_grid_dt):
    """
    Analyze and print the hyperparameter tuning results.
    
    Args:
        best_max_depth (int): Optimal max_depth value found by GridSearchCV
        param_grid_dt (dict): Parameter grid used for tuning
        
    Returns:
        str: Description of regularization level
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    print(f"\nQuestion: What was the optimal value for max_depth?")
    print(f"    Answer: {best_max_depth}")
    
    # Determine regularization level
    max_depth_values = sorted(param_grid_dt['max_depth'])
    
    if best_max_depth == max_depth_values[0]:
        regularization = "MOST regularization (shallowest tree = smallest max_depth)"
    elif best_max_depth == max_depth_values[-1]:
        regularization = "LEAST regularization (deepest tree = largest max_depth)"
    else:
        regularization = "MIDDLE regularization"
    
    print(f"\nQuestion: Did it have the most or least regularization? Or in the middle?")
    print(f"    Answer: {regularization}")
    
    # Additional interpretation
    print(f"\nInterpretation:")
    if best_max_depth < 5:
        print(f"  max_depth = {best_max_depth} -> Shallow tree (strong regularization)")
    elif best_max_depth > 5:
        print(f"  max_depth = {best_max_depth} -> Deeper tree (weak regularization)")
    else:
        print(f"  max_depth = {best_max_depth} -> Moderate tree (balanced regularization)")
    
    return regularization


def predict_test_set(gs_cv_dt, X_test, df_arrests, X_test_indices, y_test):
    """
    Make predictions on the test set and create results dataframe.
    
    Args:
        gs_cv_dt: Fitted GridSearchCV object
        X_test: Test features
        df_arrests: Original dataframe with all data
        X_test_indices: Indices of test set rows
        y_test: True labels for test set
        
    Returns:
        DataFrame: Results dataframe with columns:
            - person_id: Individual identifier
            - current_charge_felony: Feature value
            - num_fel_arrests_last_year: Feature value
            - y: True labels
            - pred_dt: Predicted probabilities
    """
    # Predict probabilities for the test set
    with suppress_stderr():
        y_pred_proba = gs_cv_dt.predict_proba(X_test)[:, 1]
    
    # Create results dataframe with ONLY what's needed
    df_results = pd.DataFrame({
        'person_id': df_arrests.loc[X_test_indices, 'person_id'].values,
        'current_charge_felony': X_test['current_charge_felony'].values,
        'num_fel_arrests_last_year': X_test['num_fel_arrests_last_year'].values,
        'y': y_test.values, 'pred_dt': y_pred_proba,})
    
    return df_results, y_pred_proba


def save_results(df_results):
    """
    Save the results dataframe to CSV.
    
    Args:
        df_results: Results dataframe with predictions
    """
    df_results.to_csv('./data/dt_predictions.csv', index=False)
    print(f"\nSaved predictions to ./data/dt_predictions.csv")
    print(f"  Shape: {df_results.shape}")
    print(f"  Columns: {list(df_results.columns)}")


def evaluate_model(gs_cv_dt, X_test, y_test, df_results):
    """
    Evaluate model performance on test set.
    
    Args:
        gs_cv_dt: Fitted GridSearchCV object
        X_test: Test features
        y_test: True labels
        df_results: Results dataframe with predictions
        
    Returns:
        tuple: (auc_score, ppv)
    """
    # Calculate and print AUC
    auc_score = gs_cv_dt.score(X_test, y_test)
    print(f"\nTest ROC-AUC: {auc_score:.4f}")
    
    # Calculate and print PPV for top 50
    top_50 = df_results.nlargest(50, 'pred_dt')
    ppv = top_50['y'].mean()
    print(f"PPV for Top 50 Highest Risk: {ppv:.4f} ({top_50['y'].sum()}/{len(top_50)})")
    
    return auc_score, ppv


def main():
    """
    Main function that orchestrates the decision tree pipeline.
    
    This function executes the complete PART 4 workflow:
        1. Load preprocessed data
        2. Create train/test split with stratification (70/30)
        3. Train decision tree with GridSearchCV for hyperparameter tuning
        4. Analyze and print hyperparameter results (optimal max_depth)
        5. Make predictions on test set
        6. Save results to CSV for PART 5
        7. Evaluate model performance (AUC and PPV)
    
    Returns:
        tuple: (df_arrests_train, df_arrests_test, df_results) where:
            - df_arrests_train: Training set with all columns
            - df_arrests_test: Test set with all columns
            - df_results: Results with predictions and features
    
    Notes:
        - Features used: 'current_charge_felony' and 'num_fel_arrests_last_year'
        - max_depth values tested: [3, 5, 7]
        - Random state set to 42 for reproducibility
        - Uses class_weight='balanced' to handle class imbalance
    """
    print("\n")
    print("="*50)
    print("DECISION TREE")
    print("="*50)
    
    # Step 1: Load data
    df_arrests = load_data()
    
    # Step 2: Create train/test split
    X_train, X_test, y_train, y_test, df_arrests_train, df_arrests_test = create_train_test_split(df_arrests)
    
    # Step 3: Train decision tree with hyperparameter tuning
    gs_cv_dt, best_max_depth, best_score = train_decision_tree(X_train, y_train)
    
    # Step 4: Analyze hyperparameter results
    param_grid_dt = {'max_depth': [3, 5, 7]}
    analyze_hyperparameter_results(best_max_depth, param_grid_dt)
    
    # Step 5: Make predictions on test set
    df_results, y_pred_proba = predict_test_set(gs_cv_dt, X_test, df_arrests, X_test.index, y_test)
    
    # Step 6: Save results
    save_results(df_results)
    
    # Step 7: Evaluate model
    auc_score, ppv = evaluate_model(gs_cv_dt, X_test, y_test, df_results)
    
    print("\n" + "="*60)
    print("DECISION TREE COMPLETED")
    print("="*60)
    
    return df_arrests_train, df_arrests_test, df_results


if __name__ == "__main__":
    main()

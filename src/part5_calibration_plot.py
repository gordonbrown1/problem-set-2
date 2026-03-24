'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def load_predictions():
    """
    Load predictions from logistic regression and decision tree models.
    
    Returns:
        tuple: (lr_df, dt_df) where:
            lr_df: DataFrame with logistic regression predictions
            dt_df: DataFrame with decision tree predictions
    """
    print("Loading predictions data-frames...")
    
    # Load logistic regression predictions
    lr_df = pd.read_csv('./data/lr_predictions.csv')
    print(f"Logistic regression predictions data-frame shape: {lr_df.shape}")
    
    # Load decision tree predictions
    dt_df = pd.read_csv('./data/dt_predictions.csv')
    print(f"Decision tree predictions data-frame shape: {dt_df.shape}")
    
    return lr_df, dt_df


def calibration_plot(y_true, y_prob, n_bins=5, title="Calibration Plot", save_path=None, show_plot=True):
    """
    Create a calibration plot with a 45-degree dashed line.
    
    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.
        title (str): Title for the plot.
        save_path (str): Path to save the plot (if None, plot is not saved).
        show_plot (bool): Whether to display the plot.
    
    Returns:
        tuple: (prob_true, bin_means) calibration curve values
    """

    # Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot the perfect calibration line (45-degree line)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
    
    # Plot the model's calibration curve
    plt.plot(bin_means, prob_true, marker='o', label="Model", linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return prob_true, bin_means


def calculate_ppv_top50(df, pred_col='pred_lr'):
    """
    Calculate Positive Predictive Value (PPV) for top 50 highest risk individuals.
    
    Parameters:
        df (DataFrame): Dataframe with true labels and predictions
        pred_col (str): Name of prediction column
        
    Returns:
        tuple: (ppv, top_50_df) where:
            - ppv (float): PPV for top 50
            - top_50_df (DataFrame): Top 50 highest risk individuals
    """

    # Sort by predicted probability (highest to lowest)
    df_sorted = df.sort_values(pred_col, ascending=False)
    
    # Take top 50
    top_50 = df_sorted.head(50)
    
    # Calculate PPV (proportion of true positives)
    ppv = top_50['y'].mean()
    
    return ppv, top_50


def calculate_auc(df, pred_col='pred_lr'):
    """
    Calculate ROC-AUC score.
    
    Parameters:
        df (DataFrame): Dataframe with true labels and predictions
        pred_col (str): Name of prediction column
        
    Returns:
        float: ROC-AUC score
    """
    auc = roc_auc_score(df['y'], df[pred_col])
    return auc


def print_comparison_results(lr_auc, dt_auc, lr_ppv, dt_ppv):
    """
    Print comparison results and answer the question about metric agreement.
    
    Parameters:
        lr_auc (float): Logistic regression AUC
        dt_auc (float): Decision tree AUC
        lr_ppv (float): Logistic regression PPV for top 50
        dt_ppv (float): Decision tree PPV for top 50
        
    Returns:
        bool: Whether both metrics agree on which model is better
    """

    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    print(f"\nLogistic Regression:")
    print(f"  AUC: {lr_auc:.4f}")
    print(f"  PPV (Top 50): {lr_ppv:.4f}")
    
    print(f"\nDecision Tree:")
    print(f"  AUC: {dt_auc:.4f}")
    print(f"  PPV (Top 50): {dt_ppv:.4f}")
    
    # Determine which model is better by each metric
    print("\n" + "="*80)
    print("QUESTION: Do both metrics agree that one model is more accurate than the other?")
    print("="*80)
    
    # AUC comparison
    if lr_auc > dt_auc:
        auc_better = "Logistic Regression"
        auc_diff = lr_auc - dt_auc
    elif dt_auc > lr_auc:
        auc_better = "Decision Tree"
        auc_diff = dt_auc - lr_auc
    else:
        auc_better = "Equal"
        auc_diff = 0
    
    # PPV comparison
    if lr_ppv > dt_ppv:
        ppv_better = "Logistic Regression"
        ppv_diff = lr_ppv - dt_ppv
    elif dt_ppv > lr_ppv:
        ppv_better = "Decision Tree"
        ppv_diff = dt_ppv - lr_ppv
    else:
        ppv_better = "Equal"
        ppv_diff = 0
    
    print(f"\nFrom AUC: {auc_better} is better (difference: {auc_diff:.4f})")
    print(f"From PPV (Top 50): {ppv_better} is better (difference: {ppv_diff:.4f})")
    
    # Check if both metrics agree
    if auc_better == ppv_better:
        agreement = "YES - Both metrics agree"
        print(f"\nAnswer: {agreement}")
        print(f"Both AUC and PPV indicate that {auc_better} is the better model.")
    else:
        agreement = "NO - Metrics disagree"
        print(f"\nAnswer: {agreement}")
        print("AUC and PPV provide different rankings of model performance.")
        print("AUC measures overall ranking ability, while PPV measures precision at the top.")
    
    return auc_better == ppv_better


def save_results_to_csv(lr_auc, dt_auc, lr_ppv, dt_ppv):
    """
    Save comparison results to CSV.
    
    Parameters:
        lr_auc (float): Logistic regression AUC
        dt_auc (float): Decision tree AUC
        lr_ppv (float): Logistic regression PPV for top 50
        dt_ppv (float): Decision tree PPV for top 50
    """

    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree'],
        'AUC': [lr_auc, dt_auc],'PPV_Top50': [lr_ppv, dt_ppv]})
    
    results_df.to_csv('./data/model_comparison.csv', index=False)
    print(f"\nSaved model comparison to ./data/model_comparison.csv\n")
    print(results_df)


def create_combined_calibration_plot(lr_df, dt_df):
    """
    Create a combined calibration plot comparing both models.
    It displays the plot and saves it uder ./data/plot
    
    Parameters:
        lr_df: Logistic regression predictions dataframe
        dt_df: Decision tree predictions dataframe
    """
    print("\n" + "="*50)
    print("CREATING COMBINED CALIBRATION PLOT")
    print("="*50)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
    
    # Calculate calibration curves for both models
    lr_prob_true, lr_prob_pred = calibration_curve(lr_df['y'], lr_df['pred_lr'], n_bins=5)
    dt_prob_true, dt_prob_pred = calibration_curve(dt_df['y'], dt_df['pred_dt'], n_bins=5)
    
    # Plot both models with axes: X = predicted, Y = actual
    plt.plot(lr_prob_pred, lr_prob_true, marker='o', label="Logistic Regression", linewidth=2, markersize=8)
    plt.plot(dt_prob_pred, dt_prob_true, marker='s', label="Decision Tree", linewidth=2, markersize=8)
    
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual)", fontsize=12)
    plt.title("Calibration Plot Comparison", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save and show
    plt.savefig('./data/plots/combined_calibration_plot.png', dpi=300, bbox_inches='tight')
    print("Saved combined calibration plot to ./data/plots/combined_calibration_plot.png")
    plt.show()


def main():
    """
    Main function that orchestrates the calibration and comparison pipeline.
    
    This function:
        1. Loads predictions from both models
        2. Creates calibration plots for both models (shows and saves)
        3. Calculates PPV for top 50 and AUC for both models
        4. Compares which model is better calibrated
        5. Determines if metrics agree on model superiority
        6. Saves all results and plots
    """
    print("\n")
    print("="*60)
    print("CALIBRATION AND MODEL COMPARISON (LR and DT)")
    print("="*60)
    
    # Create directory for plots if it doesn't exist
    if not os.path.exists('./data/plots'):
        os.makedirs('./data/plots')
        print("Created ./data/plots directory to save plots")
    
    # Load predictions
    lr_df, dt_df = load_predictions()
    
    # Create calibration plot for logistic regression
    print("\n" + "="*50)
    print("CALIBRATION PLOT: LOGISTIC REGRESSION")
    print("="*50)
    
    lr_prob_true, lr_bin_means = calibration_plot(
        y_true=lr_df['y'],y_prob=lr_df['pred_lr'],n_bins=5,
        title="Calibration Plot - Logistic Regression",
        save_path='./data/plots/lr_calibration_plot.png',
        show_plot=True)
    
    print(f"Logistic regression calibration points: {len(lr_prob_true)}")
    
    # Create calibration plot for decision tree
    print("\n" + "="*50)
    print("CALIBRATION PLOT: DECISION TREE")
    print("="*50)
    
    dt_prob_true, dt_bin_means = calibration_plot(
        y_true=dt_df['y'],
        y_prob=dt_df['pred_dt'],
        n_bins=5,
        title="Calibration Plot - Decision Tree",
        save_path='./data/plots/dt_calibration_plot.png',
        show_plot=True
    )
    print(f"Decision tree calibration points: {len(dt_prob_true)}")
    
    # Which model is more calibrated?
    print("\n" + "="*50)
    print("QUESTION: Which model is more calibrated?")
    print("="*50)
    
    # Calculate calibration error (mean absolute difference from perfect calibration)
    # For logistic regression
    lr_cal_error = np.mean(np.abs(lr_prob_true - lr_bin_means))
    print(f"\nLogistic Regression Calibration Error: {lr_cal_error:.4f}")
    
    # For decision tree
    dt_cal_error = np.mean(np.abs(dt_prob_true - dt_bin_means))
    print(f"Decision Tree Calibration Error: {dt_cal_error:.4f}")
    
    # Determine which model is better calibrated (lower error is better)
    if lr_cal_error < dt_cal_error:
        better_calibrated = "Logistic Regression"
        print(f"\nAnswer: {better_calibrated} is better calibrated (lower calibration error)")
    elif dt_cal_error < lr_cal_error:
        better_calibrated = "Decision Tree"
        print(f"\nAnswer: {better_calibrated} is better calibrated (lower calibration error)")
    else:
        better_calibrated = "Both models are equally calibrated"
        print(f"\nAnswer: {better_calibrated}")
    
    # Extra Credit - Calculate PPV for top 50
    print("\n" + "="*50)
    print("EXTRA CREDIT: PPV FOR TOP 50")
    print("="*50)
    
    lr_ppv, lr_top50 = calculate_ppv_top50(lr_df, pred_col='pred_lr')
    dt_ppv, dt_top50 = calculate_ppv_top50(dt_df, pred_col='pred_dt')
    
    print(f"\nLogistic Regression PPV (Top 50): {lr_ppv:.4f} ({lr_top50['y'].sum()}/{len(lr_top50)})")
    print(f"Decision Tree PPV (Top 50): {dt_ppv:.4f} ({dt_top50['y'].sum()}/{len(dt_top50)})")
    
    # Extra Credit - Calculate AUC
    print("\n" + "="*50)
    print("EXTRA CREDIT: AUC")
    print("="*50)
    
    lr_auc = calculate_auc(lr_df, pred_col='pred_lr')
    dt_auc = calculate_auc(dt_df, pred_col='pred_dt')
    
    print(f"\nLogistic Regression AUC: {lr_auc:.4f}")
    print(f"Decision Tree AUC: {dt_auc:.4f}")
    
    # Extra Credit - Do metrics agree?
    agreement = print_comparison_results(lr_auc, dt_auc, lr_ppv, dt_ppv)
    
    # Save results to CSV
    save_results_to_csv(lr_auc, dt_auc, lr_ppv, dt_ppv)
    
    # Create combined calibration plot (shows and saves)
    create_combined_calibration_plot(lr_df, dt_df)
    
    # Print summary
    print("\n" + "="*60)
    print("CALIBRATION AND MODEL COMPARISON COMPLETE")
    print("="*60)
    print("\nFiles saved to:")
    print("  ./data/plots/lr_calibration_plot.png")
    print("  ./data/plots/dt_calibration_plot.png")
    print("  ./data/plots/combined_calibration_plot.png")
    print("  ./data/model_comparison.csv\n")
    
    return lr_df, dt_df


if __name__ == "__main__":
    main()




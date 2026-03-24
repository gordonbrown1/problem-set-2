'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np
from datetime import timedelta


# Your code here

def load_data():
    """
    Load the raw data from part1_etl.
    
    Returns:
        tuple: (pred_universe_raw, arrest_events_raw) dataframes
    """

    print("\nLoading data after part1_etl...")
    pred_universe_raw = pd.read_csv('./data/pred_universe_raw.csv')
    arrest_events_raw = pd.read_csv('./data/arrest_events_raw.csv')
    
    # Ensure date columns are datetime
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['arrest_date_univ'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'])
    
    #print(f"Loaded pred_universe_raw: {pred_universe_raw.shape}")
    #print(f"Loaded arrest_events_raw: {arrest_events_raw.shape}")
    
    return pred_universe_raw, arrest_events_raw


def merge_on_person_id(pred_universe_raw, arrest_events_raw):
    """
    Perform a full outer join on 'person_id' as instructed by the professor.
    
    Note: This will create a Cartesian product - each person in pred_universe
    will match with all their charges from arrest_events.
    
    Args:
        pred_universe_raw: Dataframe with all arrests
        arrest_events_raw: Dataframe with all charges
        
    Returns:
        DataFrame: Merged dataframe with all person-arrest combinations
    """
    print("\n" + "="*50)
    print("MERGING ON PERSON_ID")
    print("="*50)
    
    # Perform full outer join on person_id only
    df_arrests = pd.merge(pred_universe_raw,arrest_events_raw,
        on='person_id',how='outer',suffixes=('_univ', '_event'),
        indicator=True)
    
    print(f"Merged dataframe shape: {df_arrests.shape}")
    
    return df_arrests


def create_target_variable(df_arrests, arrest_events_raw):
    """
    Create target variable y: 1 if person was arrested for a felony in the 365 days AFTER arrest date.
    
    For each arrest in df_arrests, check if the same person appears in arrest_events_raw
    with a felony charge between (arrest_date + 1 day) and (arrest_date + 365 days).
    
    Args:
        df_arrests: Merged dataframe
        arrest_events_raw: Original arrest events dataframe
        
    Returns:
        DataFrame: df_arrests with new 'y' column
    """
    print("\n" + "="*60)
    print("CREATING TARGET VARIABLE: Felony rearrest within 365 days")
    print("="*60)
    
    # Create a copy to avoid warnings
    df = df_arrests.copy()
    
    # Initialize y column to 0
    df['y'] = 0
    
    # Convert arrest_date_event in arrest_events_raw to datetime if not already
    events = arrest_events_raw.copy()
    events['arrest_date_event'] = pd.to_datetime(events['arrest_date_event'])
    
    # For each row in df_arrests
    for idx, row in df.iterrows():
        # Only proceed if we have an arrest date from the universe table
        if pd.notna(row['arrest_date_univ']):
            person_id = row['person_id']
            arrest_date = row['arrest_date_univ']
            
            # Define the 365-day window after arrest (starting next day)
            start_date = arrest_date + timedelta(days=1)
            end_date = arrest_date + timedelta(days=365)
            
            # Check if this person has any felony arrests in events within this window
            # Note: Using 'charge_degree' instead of 'degree'
            person_felony_rearrests = events[
                (events['person_id'] == person_id) &
                (events['arrest_date_event'] >= start_date) &
                (events['arrest_date_event'] <= end_date) &
                (events['charge_degree'].str.lower() == 'felony')
            ]
            
            if len(person_felony_rearrests) > 0:
                df.loc[idx, 'y'] = 1
    
    # Calculate and print the share
    share_rearrested = df['y'].mean()
    print(f"\nQuestion: What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year?")
    print(f"Answer: {share_rearrested:.4f} ({share_rearrested:.2%})")
    
    return df


def create_current_charge_felony(df_arrests):
    """
    Create predictive feature current_charge_felony: 1 if current arrest was for a felony charge.
    
    This uses the 'charge_degree' column from arrest_events to determine if the current charge is a felony.
    For rows that don't have a charge_degree (uncharged arrests), this will be 0.
    
    Args:
        df_arrests: Dataframe with merged data
        
    Returns:
        DataFrame: df_arrests with new 'current_charge_felony' column
    """

    print("\n" + "="*50)
    print("CREATING FEATURE: 'current_charge_felony'")
    print("="*50)
    
    df = df_arrests.copy()
    df['current_charge_felony'] = (df['charge_degree'].str.lower() == 'felony').fillna(0).astype(int)
    
    # Calculate and print the share
    share_felony = df['current_charge_felony'].mean()
    print(f"\nQuestion: What share of current charges are felonies?")
    print(f"Answer: {share_felony:.4f} ({share_felony:.2%})")
    
    return df



def create_num_fel_arrests_last_year(df_arrests, arrest_events_raw):
    """
    Create predictive feature num_fel_arrests_last_year: number of felony arrests in the year PRIOR.
    
    For each arrest, count how many felony arrests the person had in the 365 days before
    the current arrest date (from arrest_date_univ).
    
    Args:
        df_arrests: Dataframe with merged data
        arrest_events_raw: Original arrest events dataframe
        
    Returns:
        DataFrame: df_arrests with new 'num_fel_arrests_last_year' column
    """
    print("\n" + "="*50)
    print("CREATING FEATURE: 'num_fel_arrests_last_year'")
    print("="*50)
    
    df = df_arrests.copy()
    events = arrest_events_raw.copy()
    events['arrest_date_event'] = pd.to_datetime(events['arrest_date_event'])
    
    # Initialize column
    df['num_fel_arrests_last_year'] = 0
    
    # For each row in df_arrests
    for idx, row in df.iterrows():
        if pd.notna(row['arrest_date_univ']):
            person_id = row['person_id']
            arrest_date = row['arrest_date_univ']
            
            # Define the 365-day window prior to the arrest
            start_date = arrest_date - timedelta(days=365)
            end_date = arrest_date  # Up to but not including current arrest
            
            # Filter all felony arrests for this person in the prior year
            prior_felonies = events[
                (events['person_id'] == person_id) &
                (events['arrest_date_event'] >= start_date) &
                (events['arrest_date_event'] < end_date) &
                (events['charge_degree'].str.lower() == 'felony')]
            
            # Count unique arrests (not charges)
            num_unique = prior_felonies['arrest_id'].nunique()
            df.loc[idx, 'num_fel_arrests_last_year'] = num_unique
    
    # Calculate and print the average
    avg_fel_arrests = df['num_fel_arrests_last_year'].mean()
    print(f"\nQuestion: What is the average number of felony arrests in the last year?")
    print(f"Answer: {avg_fel_arrests:.4f}")
    
    # Print the mean as requested
    print(f"\nMean of 'num_fel_arrests_last_year': {avg_fel_arrests:.4f}")
    
    return df


def print_pred_universe_head(df_arrests):
    """
    Print the head of the pred_universe portion as requested.
    
    Args:
        df_arrests: Merged dataframe
    """
    print("\n" + "="*50)
    print("pred_universe.head()")
    print("="*50)
    
    # Get columns that are from pred_universe (ending with _univ or original names)
    univ_columns = [col for col in df_arrests.columns if col.endswith('_univ') or 
                    col in ['person_id', 'age_at_arrest', 'sex', 'race', 'arrest_date_univ']]
    
    # Show first 5 rows of the pred_universe data
    print(df_arrests[univ_columns].head())
    
    # Also show the key columns for context
    print("\n" + "="*77)
    print("Key columns: person_id, current_charge_felony, num_fel_arrests_last_year, y:")
    print("="*77)
    print(df_arrests[['person_id', 'current_charge_felony', 'num_fel_arrests_last_year', 'y']].head())


def save_preprocessed_data(df_arrests):
    """
    Save the preprocessed dataframe for use in ML.
    
    Args:
        df_arrests: Final preprocessed dataframe
    
        
    Side effect:
        Saves preprocessed dataframe to ./data/df_arrests.csv and prints the shape

    """
    import os
    # Create data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    # Save to CSV for use in PART 3
    df_arrests.to_csv('./data/df_arrests.csv', index=False)
    print(f"\nPreprocessed data file saved to ./data/df_arrests.csv")
    print(f"  Shape: {df_arrests.shape}")
    #print(f"  Columns: {list(df_arrests.columns)}")


def main():
    """
    Main function that orchestrates the preprocessing pipeline.
    
    Steps:
    1. Load data from part1_etl
    2. Perform full outer join on person_id (as instructed by professor)
    3. Create target variable (y) - felony rearrest within 365 days after arrest
    4. Create feature: current_charge_felony
    5. Create feature: num_fel_arrests_last_year
    6. Print required statistics
    7. Print pred_universe.head()
    8. Save preprocessed data for PART 3
    """
    
    # Step 1: Load data
    pred_universe_raw, arrest_events_raw = load_data()
    
    # Step 2: Merge on person_id (as instructed by professor)
    df_arrests = merge_on_person_id(pred_universe_raw, arrest_events_raw)
    
    # Step 3: Create target variable (y) - rearrest for felony within 365 days
    df_arrests = create_target_variable(df_arrests, arrest_events_raw)
    
    # Step 4: Create current_charge_felony feature
    df_arrests = create_current_charge_felony(df_arrests)
    
    # Step 5: Create num_fel_arrests_last_year feature
    df_arrests = create_num_fel_arrests_last_year(df_arrests, arrest_events_raw)
    
    # Step 6: Print pred_universe.head() as requested
    print_pred_universe_head(df_arrests)
    
    # Step 7: Save preprocessed data
    save_preprocessed_data(df_arrests)
    
    print("\n" + "="*50)
    print("PRE-PROCESSING COMPLETED")
    print("="*50)
    
    # Return df_arrests for use in main.py
    return df_arrests


if __name__ == "__main__":
    main()
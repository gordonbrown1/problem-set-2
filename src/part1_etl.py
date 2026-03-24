'''
PART 1: ETL the two datasets and save each in a folder called `data/` as .csv's
'''


# source venv/bin/activate

import pandas as pd
import os

pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
pred_universe_raw.drop(columns=['filing_date'], inplace=True)
arrest_events_raw.drop(columns=['filing_date'], inplace=True)

# Inject and save both data frames to `data/` -> 'pred_universe_raw.csv', 'arrest_events_raw.csv'

def save_data():
    """
    This function creates the ./data directory if it doesn't exist,
    then saves the transformed pred_universe_raw and arrest_events_raw
    dataframes to CSV format.
    
    Returns:
    None
    
    Side Effects:
        Creates ./data directory if it doesn't exist
        Writes two CSV files to the ./data directory
        Prints status messages about the saved files
    """

    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Created ./data/ directory")

    # Save to CSV
    pred_universe_raw.to_csv('./data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('./data/arrest_events_raw.csv', index=False)

    print(f"Saved pred_universe_raw.csv with {len(pred_universe_raw)} rows")
    print(f"Saved pred_universe_raw.csv with {len(arrest_events_raw)} rows")


    

    




def main():
    """
    This function orchestrates the ETL pipeline by:
        Printing initial data shapes
        Calling save_data() to persist the transformed dataframes
    
    Returns:
    None
    
    Side Effects:
        Prints progress information to console
        Triggers file writing via save_data()
    """
    print("\nStarting ETL/saving process...\n")
    print(f"pred_universe_raw shape: {pred_universe_raw.shape}")
    print(f"arrest_events_raw shape: {arrest_events_raw.shape}\n")

    # Save the data
    save_data()
    
    print("\nETL/save Completed\n")

    print(f"pred_universe_raw: \n{pred_universe_raw.head()}\n\n")
    print(f"arrest_events_raw: \n{arrest_events_raw.head()}")


# Code below is used to test the script directly
if __name__ == "__main__":
    main()
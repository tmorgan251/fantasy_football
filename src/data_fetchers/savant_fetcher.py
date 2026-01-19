import pandas as pd
import os

def load_savant_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Download CSVs from nflsavant.com first.")
    
    df = pd.read_csv(filepath)
    return df

def filter_red_zone_plays(df):
    return df[df['YardLine'] <= 20]

def get_target_counts(df):
    targets = df[df['PassAttempt'] == 1]
    return targets['Receiver'].value_counts()

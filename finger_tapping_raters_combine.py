import pandas as pd
import os

original_folder = r'\\files.ubc.ca\team\PPRC\Camera\Booth_Results\finger_tapping_ws\Experiment2\csvs'
dest_folder = r'\\files.ubc.ca\team\PPRC\CAMERA\Booth_Processed\hand_movement\docs'
files = ['CAMERA Study Booth - Tracking Log_KW-No Names.csv', 
         'CAMERA Study Booth - Tracking Log_MG.csv', 
         'CAMERA Study Booth - Tracking Log_SA.csv',
         'CAMERA Study Booth - Tracking Log_WM.csv',
         'CAMERA Study Booth - Tracking Log_TM.csv',]
suffixes = ['_KW', '_MG', '_SA', '_WM','_TM']

# Initialize an empty DataFrame to store combined results
combined_df = None
right_column = 'Right_Postural_tremor_UPDRS'
left_column = 'Left_Postural_tremor_UPDRS'

# Loop through the files and suffixes
for file, suffix in zip(files, suffixes):
    # Read the CSV file with proper encoding
    df = pd.read_csv(os.path.join(original_folder, file), encoding='ISO-8859-1')
    
    # Set the second row as header
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    # Ensure ID and Date columns are present
    if 'ID' not in df.columns or 'Date' not in df.columns:
        raise ValueError(f"ID and Date columns must be present in the file: {file}")
    
    # Extract the relevant columns (Right_Collection_UPDRS and Left_Collection_UPDRS)
    df = df[['ID', 'Date', right_column, left_column]]
    
    # Rename the columns to include the suffix
    df = df.rename(columns={
        right_column: right_column + suffix,
        left_column: left_column + suffix
    })
    # set the id and date as sring
    df['ID'] = df['ID'].astype(str)
    # Change the date from y-m-d to YYYYMMDD
    df['Date'] = pd.to_datetime(df['Date'], format="mixed").dt.strftime('%Y%m%d')
    
    # Merge the DataFrame with the combined DataFrame based on ID and Date
    if combined_df is None:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on=['ID', 'Date'], how='outer')


# Save the combined DataFrame to a CSV file
combined_df.to_csv(os.path.join(dest_folder, 'postural_tremor.csv'), index=False)
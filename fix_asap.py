"""
Fix ASAP dataset column names to match expected format
"""

import pandas as pd

# Load the ASAP dataset
print("Loading ASAP dataset...")
df = pd.read_csv('data/asap_training.csv')

print(f"Original columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")

# Rename columns to expected format
df = df.rename(columns={
    'full_text': 'text',  # Rename to 'text'
    'assignment': 'prompt_id'  # Rename assignment to prompt_id
})

# Keep only necessary columns for training
columns_to_keep = ['essay_id', 'text', 'score', 'prompt_id', 
                   'economically_disadvantaged', 'student_disability_status', 
                   'ell_status', 'race_ethnicity', 'gender']

# Check which columns exist
available_cols = [col for col in columns_to_keep if col in df.columns]
df_clean = df[available_cols]

print(f"\nNew columns: {list(df_clean.columns)}")
print(f"Rows after cleaning: {len(df_clean)}")

# Show score distribution
print(f"\nScore distribution:")
print(df_clean['score'].value_counts().sort_index())

# Show prompt distribution
if 'prompt_id' in df_clean.columns:
    print(f"\nPrompt distribution:")
    print(df_clean['prompt_id'].value_counts().sort_index())

# Save the fixed dataset
output_path = 'data/asap_training_fixed.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n[SUCCESS] Fixed dataset saved to: {output_path}")

# Show sample
print(f"\nSample essay (first 200 chars):")
print(df_clean['text'].iloc[0][:200] + "...")
"""
Dataset Validation Script for AES Project - Windows Compatible
Checks if your datasets are in the correct format

Usage: python validate_datasets.py
"""

import os
import sys
import pandas as pd
import zipfile
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def validate_file_exists(path):
    """Check if file exists"""
    if os.path.exists(path):
        print(f"  [OK] File found: {path}")
        return True
    else:
        print(f"  [ERROR] File NOT found: {path}")
        return False

def validate_asap(file_path):
    """Validate ASAP dataset"""
    print("\n" + "="*60)
    print("VALIDATING ASAP DATASET")
    print("="*60)
    
    if not validate_file_exists(file_path):
        return False
    
    try:
        # Try reading as CSV
        try:
            df = pd.read_csv(file_path)
        except:
            # Try as TSV
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
        
        print(f"  [OK] Successfully loaded: {len(df)} rows")
        print(f"\n  Columns found: {list(df.columns)}")
        
        # Check required columns
        required_cols = {
            'essay': ['essay', 'text', 'essay_text'],
            'score': ['domain1_score', 'rater1_domain1', 'score'],
            'prompt': ['essay_set', 'prompt_id', 'set']
        }
        
        found_text = False
        found_score = False
        found_prompt = False
        
        # Check for text column
        for col in required_cols['essay']:
            if col in df.columns:
                print(f"  [OK] Text column found: '{col}'")
                found_text = True
                text_col = col
                break
        
        if not found_text:
            print(f"  [ERROR] No text column found. Need one of: {required_cols['essay']}")
            return False
        
        # Check for score column
        for col in required_cols['score']:
            if col in df.columns:
                print(f"  [OK] Score column found: '{col}'")
                found_score = True
                score_col = col
                break
        
        if not found_score:
            print(f"  [ERROR] No score column found. Need one of: {required_cols['score']}")
            return False
        
        # Check for prompt column (optional but recommended)
        for col in required_cols['prompt']:
            if col in df.columns:
                print(f"  [OK] Prompt column found: '{col}'")
                found_prompt = True
                prompt_col = col
                break
        
        if not found_prompt:
            print(f"  [WARNING] No prompt column found. Recommended: {required_cols['prompt']}")
        
        # Show statistics
        print(f"\n  Dataset Statistics:")
        print(f"    Total essays: {len(df)}")
        print(f"    Score range: {df[score_col].min()} to {df[score_col].max()}")
        print(f"    Missing scores: {df[score_col].isnull().sum()}")
        
        if found_prompt:
            print(f"    Number of prompts: {df[prompt_col].nunique()}")
            print(f"    Essays per prompt:")
            print(df[prompt_col].value_counts().sort_index().to_string())
        
        # Check essay content
        sample_essay = str(df[text_col].iloc[0])
        print(f"\n  Sample essay (first 200 chars):")
        print(f"    {sample_essay[:200]}...")
        
        avg_length = df[text_col].astype(str).str.split().str.len().mean()
        print(f"\n  Average essay length: {avg_length:.0f} words")
        
        print("\n  [SUCCESS] ASAP dataset is VALID and ready to use!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error reading ASAP dataset: {e}")
        return False

def validate_toefl11(file_path):
    """Validate TOEFL11 dataset"""
    print("\n" + "="*60)
    print("VALIDATING TOEFL11 DATASET")
    print("="*60)
    
    if not validate_file_exists(file_path):
        return False
    
    try:
        df = pd.read_csv(file_path)
        print(f"  [OK] Successfully loaded: {len(df)} rows")
        print(f"\n  Columns found: {list(df.columns)}")
        
        # Check required columns
        required_text = ['text', 'essay', 'full_text']
        required_score = ['score', 'rating', 'grade']
        required_l1 = ['L1', 'native_lang', 'native_language', 'l1']
        
        found_text = any(col in df.columns for col in required_text)
        found_score = any(col in df.columns for col in required_score)
        found_l1 = any(col in df.columns for col in required_l1)
        
        if found_text:
            text_col = next(col for col in required_text if col in df.columns)
            print(f"  [OK] Text column found: '{text_col}'")
        else:
            print(f"  [ERROR] No text column found. Need one of: {required_text}")
            return False
        
        if found_score:
            score_col = next(col for col in required_score if col in df.columns)
            print(f"  [OK] Score column found: '{score_col}'")
        else:
            print(f"  [ERROR] No score column found. Need one of: {required_score}")
            return False
        
        if found_l1:
            l1_col = next(col for col in required_l1 if col in df.columns)
            print(f"  [OK] Native language column found: '{l1_col}'")
        else:
            print(f"  [WARNING] No L1 column found. Recommended: {required_l1}")
        
        # Statistics
        print(f"\n  Dataset Statistics:")
        print(f"    Total essays: {len(df)}")
        print(f"    Score range: {df[score_col].min()} to {df[score_col].max()}")
        
        if found_l1:
            print(f"    Native languages: {df[l1_col].nunique()}")
            print(f"    Languages:")
            print(df[l1_col].value_counts().to_string())
        
        avg_length = df[text_col].astype(str).str.split().str.len().mean()
        print(f"\n  Average essay length: {avg_length:.0f} words")
        
        print("\n  [SUCCESS] TOEFL11 dataset is VALID and ready to use!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error reading TOEFL11 dataset: {e}")
        return False

def validate_feedback_prize(file_path):
    """Validate Feedback Prize dataset"""
    print("\n" + "="*60)
    print("VALIDATING FEEDBACK PRIZE DATASET")
    print("="*60)
    
    if not validate_file_exists(file_path):
        return False
    
    try:
        # Handle ZIP files
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_files:
                    print(f"  [ERROR] No CSV files found in ZIP")
                    return False
                print(f"  [OK] Found {len(csv_files)} CSV file(s) in ZIP")
                csv_name = csv_files[0]
                df = pd.read_csv(z.open(csv_name))
        else:
            df = pd.read_csv(file_path)
        
        print(f"  [OK] Successfully loaded: {len(df)} rows")
        print(f"\n  Columns found: {list(df.columns)}")
        
        # Check required columns
        required_text = ['full_text', 'text', 'essay']
        score_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        
        found_text = any(col in df.columns for col in required_text)
        found_scores = sum(col in df.columns for col in score_cols)
        
        if found_text:
            text_col = next(col for col in required_text if col in df.columns)
            print(f"  [OK] Text column found: '{text_col}'")
        else:
            print(f"  [ERROR] No text column found. Need one of: {required_text}")
            return False
        
        if found_scores > 0:
            print(f"  [OK] Found {found_scores} score columns:")
            for col in score_cols:
                if col in df.columns:
                    print(f"      - {col}")
        else:
            print(f"  [ERROR] No score columns found. Need at least one of: {score_cols}")
            return False
        
        # Statistics
        print(f"\n  Dataset Statistics:")
        print(f"    Total essays: {len(df)}")
        
        for col in score_cols:
            if col in df.columns:
                print(f"    {col}: range {df[col].min():.1f} to {df[col].max():.1f}")
        
        avg_length = df[text_col].astype(str).str.split().str.len().mean()
        print(f"\n  Average essay length: {avg_length:.0f} words")
        
        print("\n  [SUCCESS] Feedback Prize dataset is VALID and ready to use!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error reading Feedback Prize dataset: {e}")
        return False

def scan_directory(directory='data'):
    """Scan directory for potential datasets"""
    print("\n" + "="*60)
    print(f"SCANNING DIRECTORY: {directory}")
    print("="*60)
    
    if not os.path.exists(directory):
        print(f"  [ERROR] Directory does not exist: {directory}")
        print(f"\n  Creating directory: {directory}")
        os.makedirs(directory)
        print(f"  [OK] Directory created: {directory}")
        return None
    
    files = list(Path(directory).glob('**/*'))
    csv_files = [f for f in files if f.suffix.lower() in ['.csv', '.tsv', '.zip']]
    
    if not csv_files:
        print(f"  [ERROR] No CSV, TSV, or ZIP files found in {directory}")
        return None
    
    print(f"\n  Found {len(csv_files)} potential dataset file(s):")
    for i, f in enumerate(csv_files, 1):
        print(f"    {i}. {f}")
    
    return csv_files

def main():
    """Main validation function"""
    print("\n" + "="*80)
    print("AES DATASET VALIDATION TOOL")
    print("="*80)
    print("\nThis tool will check if your datasets are in the correct format.")
    
    # Scan for files
    files = scan_directory('data')
    
    if not files:
        print("\n" + "="*80)
        print("INSTRUCTIONS:")
        print("="*80)
        print("1. Place your datasets in the 'data' folder:")
        print("   - asap_training.csv (or .tsv)")
        print("   - toefl11_train.csv")
        print("   - feedback_prize_train.csv (or .zip)")
        print("2. Run this script again")
        return
    
    # Try to identify and validate datasets
    print("\n" + "="*80)
    print("AUTOMATIC VALIDATION")
    print("="*80)
    
    validated = []
    
    for file_path in files:
        file_name = file_path.name.lower()
        
        # Try to identify dataset type
        if 'asap' in file_name or 'essay_set' in file_name:
            if validate_asap(str(file_path)):
                validated.append(('ASAP', str(file_path)))
        
        elif 'toefl' in file_name or 'l1' in file_name:
            if validate_toefl11(str(file_path)):
                validated.append(('TOEFL11', str(file_path)))
        
        elif 'feedback' in file_name or 'prize' in file_name:
            if validate_feedback_prize(str(file_path)):
                validated.append(('Feedback Prize', str(file_path)))
        
        else:
            # Try all validators
            print(f"\n  Trying to identify: {file_path}")
            if validate_asap(str(file_path)):
                validated.append(('ASAP', str(file_path)))
            elif validate_toefl11(str(file_path)):
                validated.append(('TOEFL11', str(file_path)))
            elif validate_feedback_prize(str(file_path)):
                validated.append(('Feedback Prize', str(file_path)))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if validated:
        print(f"\n[SUCCESS] Validated {len(validated)} dataset(s):")
        for dataset_type, path in validated:
            print(f"  - {dataset_type}: {path}")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\nYou can now train your model with:")
        
        if len(validated) == 1:
            print(f"\npython aes_complete.py \\")
            print(f"  --train_csvs \"{validated[0][1]}\" \\")
            print(f"  --model_type transformer \\")
            print(f"  --epochs 3 \\")
            print(f"  --do_fairness")
        
        else:
            train_sets = [f'"{v[1]}"' for v in validated[:2]]
            test_sets = [f'"{v[1]}"' for v in validated[2:]] if len(validated) > 2 else []
            
            print(f"\npython aes_complete.py \\")
            print(f"  --train_csvs {' '.join(train_sets)} \\")
            if test_sets:
                print(f"  --test_csvs {' '.join(test_sets)} \\")
            print(f"  --model_type transformer \\")
            print(f"  --epochs 3 \\")
            print(f"  --do_fairness")
    
    else:
        print("\n[ERROR] No valid datasets found!")
        print("\nPlease check that your files have the correct format.")
        print("See the error messages above for specific issues.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
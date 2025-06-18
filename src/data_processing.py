"""
Data processing module for surname correction project.
Handles loading and preprocessing data from Excel files and transcriptions.
"""

import os
import pandas as pd
import re
from typing import List, Tuple, Dict, Any


def load_reference_data(file_path: str) -> pd.DataFrame:
    """
    Load the reference data containing correct names and surnames from Excel or CSV.
    
    Args:
        file_path: Path to the file with correct names and surnames (Excel or CSV)
        
    Returns:
        DataFrame with the reference data
    """
    try:
        # Check file extension and use appropriate pandas function
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        print(f"Successfully loaded reference data with {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error loading reference data: {e}")
        return pd.DataFrame()


def extract_names_surnames(df: pd.DataFrame, name_col: str = "full_name") -> Tuple[List[str], List[str]]:
    """
    Extract names and surnames from a DataFrame column containing full names.
    
    Args:
        df: DataFrame with the reference data
        name_col: Column name containing the full names
        
    Returns:
        Tuple of (names list, surnames list)
    """
    if name_col not in df.columns:
        raise ValueError(f"Column '{name_col}' not found in DataFrame")
    
    names = []
    surnames = []
    
    for full_name in df[name_col]:
        if not isinstance(full_name, str):
            continue
            
        parts = full_name.strip().split()
        if len(parts) >= 2:
            # Assuming the last word(s) are surnames
            # This is a simplification - actual logic may need to be adapted to your naming conventions
            name = parts[0]
            surname = " ".join(parts[1:])
            names.append(name)
            surnames.append(surname)
    
    return names, surnames


def load_transcriptions(directory: str) -> List[str]:
    """
    Load transcription text files from a directory.
    
    Args:
        directory: Path to directory containing transcription files
        
    Returns:
        List of transcription texts
    """
    transcriptions = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcriptions.append(f.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return transcriptions


def extract_potential_surnames(text: str) -> List[str]:
    """
    Extract potential surnames from transcription text.
    This is a simple implementation that can be enhanced with NER or other techniques.
    
    Args:
        text: Transcription text
        
    Returns:
        List of potential surnames
    """
    # Simple extraction based on capitalized words
    # In a real implementation, you might use NER or more sophisticated techniques
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    return words


def create_training_data(correct_surnames: List[str], 
                         transcriptions: List[str], 
                         augment: bool = True) -> Tuple[List[str], List[str]]:
    """
    Create training data pairs of (misspelled surname, correct surname).
    
    Args:
        correct_surnames: List of correct surnames
        transcriptions: List of transcription texts
        augment: Whether to augment data with synthetic misspellings
        
    Returns:
        Tuple of (misspelled surnames, correct surnames)
    """
    misspelled = []
    correct = []
    
    # Extract potential surnames from transcriptions
    potential_surnames = []
    for text in transcriptions:
        potential_surnames.extend(extract_potential_surnames(text))
    
    # Create a dictionary of correct surnames for lookup
    surname_dict = {s.lower(): s for s in correct_surnames}
    
    # Find potential misspellings in transcriptions
    for surname in potential_surnames:
        # Check if this is a misspelling of a known surname
        # In a real implementation, you would use more sophisticated matching
        for correct_surname in correct_surnames:
            if surname.lower() != correct_surname.lower() and are_similar(surname, correct_surname):
                misspelled.append(surname)
                correct.append(correct_surname)
                break
    
    # Augment with synthetic misspellings if requested
    if augment and correct_surnames:
        from src.feature_engineering import generate_misspellings
        
        for surname in correct_surnames[:min(100, len(correct_surnames))]:  # Limit to avoid too much data
            synthetic_misspellings = generate_misspellings(surname, num=3)
            misspelled.extend(synthetic_misspellings)
            correct.extend([surname] * len(synthetic_misspellings))
    
    return misspelled, correct


def are_similar(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """
    Check if two strings are similar using Levenshtein distance.
    
    Args:
        s1: First string
        s2: Second string
        threshold: Similarity threshold (0-1)
        
    Returns:
        True if strings are similar, False otherwise
    """
    from fuzzywuzzy import fuzz
    
    # Calculate similarity ratio
    ratio = fuzz.ratio(s1.lower(), s2.lower()) / 100.0
    
    return ratio >= threshold


def split_data(X: List[Any], y: List[Any], test_size: float = 0.2) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

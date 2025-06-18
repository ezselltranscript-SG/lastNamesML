"""
Script to preprocess the Cleaned_Surnames.csv file and prepare it for training.
"""

import os
import pandas as pd
import re
import random
from typing import List, Tuple


def clean_surname(text: str) -> str:
    """
    Clean a surname entry by removing quotes and extracting only the surname part.
    
    Args:
        text: Raw surname text from CSV
        
    Returns:
        Cleaned surname
    """
    # Remove quotes
    text = text.strip('"')
    
    # If there's a comma, take the part after the comma (the surname)
    if ',' in text:
        parts = text.split(',')
        if len(parts) >= 2:
            return parts[1].strip()
    
    return text


def preprocess_surnames_csv(csv_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess the surnames CSV file.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path to save the processed CSV file
        
    Returns:
        Processed DataFrame
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if the expected column exists
    if 'ExtractedSurname' not in df.columns:
        raise ValueError("CSV file does not have 'ExtractedSurname' column")
    
    # Clean surnames
    df['CleanedSurname'] = df['ExtractedSurname'].apply(clean_surname)
    
    # Remove empty or very short surnames
    df = df[df['CleanedSurname'].str.len() > 2]
    
    # Create a full name column (for compatibility with the existing code)
    # Using a placeholder first name "John" for simplicity
    df['full_name'] = "John " + df['CleanedSurname']
    
    # Save the processed DataFrame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def generate_simple_misspellings(word: str, num: int = 1) -> List[str]:
    """
    Generate simple misspellings of a word.
    
    Args:
        word: Original word
        num: Number of misspellings to generate
        
    Returns:
        List of misspelled versions
    """
    if not word or len(word) < 3:
        return [word]
    
    misspellings = []
    operations = ['substitute', 'delete', 'insert', 'swap']
    
    for _ in range(num):
        misspelled = word
        operation = random.choice(operations)
        
        if operation == 'substitute' and len(misspelled) > 0:
            # Substitute a random character
            pos = random.randint(0, len(misspelled) - 1)
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            misspelled = misspelled[:pos] + char + misspelled[pos+1:]
            
        elif operation == 'delete' and len(misspelled) > 3:
            # Delete a random character
            pos = random.randint(0, len(misspelled) - 1)
            misspelled = misspelled[:pos] + misspelled[pos+1:]
            
        elif operation == 'insert':
            # Insert a random character
            pos = random.randint(0, len(misspelled))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            misspelled = misspelled[:pos] + char + misspelled[pos:]
            
        elif operation == 'swap' and len(misspelled) > 1:
            # Swap two adjacent characters
            pos = random.randint(0, len(misspelled) - 2)
            misspelled = misspelled[:pos] + misspelled[pos+1] + misspelled[pos] + misspelled[pos+2:]
        
        if misspelled != word:
            misspellings.append(misspelled)
        else:
            misspellings.append(word + 'x')  # Fallback
    
    return misspellings


def generate_misspelled_transcriptions(surnames: List[str], 
                                      output_dir: str, 
                                      num_files: int = 5,
                                      misspellings_per_file: int = 10) -> None:
    """
    Generate sample transcription files with misspelled surnames.
    
    Args:
        surnames: List of correct surnames
        output_dir: Directory to save the transcription files
        num_files: Number of transcription files to generate
        misspellings_per_file: Number of misspelled surnames per file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate transcription files
    for i in range(num_files):
        # Select random surnames
        selected_surnames = random.sample(
            surnames, 
            min(misspellings_per_file, len(surnames))
        )
        
        # Generate misspellings
        misspelled_text = ""
        
        for surname in selected_surnames:
            # Generate a misspelled version
            misspelled = generate_simple_misspellings(surname, num=1)[0]
            
            # Add to text with some context
            misspelled_text += f"The document was signed by Mr. {misspelled}.\n"
            misspelled_text += f"We received a letter from Mrs. {generate_simple_misspellings(random.choice(surnames), num=1)[0]}.\n"
        
        # Save to file
        output_path = os.path.join(output_dir, f"transcription_{i+1}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(misspelled_text)


def main():
    """Main function to preprocess data."""
    # Paths
    input_csv = "Cleaned_Surnames.csv"
    output_csv = "data/processed_surnames.csv"
    transcriptions_dir = "data/transcriptions"
    
    # Preprocess CSV
    print(f"Preprocessing {input_csv}")
    df = preprocess_surnames_csv(input_csv, output_csv)
    print(f"Processed {len(df)} surnames")
    print(f"Saved to {output_csv}")
    
    # Get list of clean surnames
    surnames = df['CleanedSurname'].tolist()
    
    # Generate misspelled transcriptions
    print(f"Generating sample transcriptions in {transcriptions_dir}")
    generate_misspelled_transcriptions(surnames, transcriptions_dir)
    print(f"Generated sample transcriptions")
    
    print("Data preprocessing complete!")


if __name__ == '__main__':
    main()

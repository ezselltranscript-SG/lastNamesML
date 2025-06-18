"""
Script to correct surnames in transcribed text using the trained model.
"""

import os
import argparse
import pandas as pd
import pickle
from typing import List, Dict, Any, Tuple

from src.models import SurnameCorrector
from src.data_processing import load_reference_data, extract_names_surnames


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Correct surnames in transcribed text')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file or directory with transcriptions')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save corrected transcriptions')
    parser.add_argument('--model', type=str, default='models/random_forest_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--reference-data', type=str,
                        help='Path to Excel file with reference names and surnames')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Minimum confidence threshold for applying corrections')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'svm', 'neural_network'],
                        help='Type of model to use')
    parser.add_argument('--report', action='store_true',
                        help='Generate correction report')
    
    return parser.parse_args()


def load_model(model_path: str, model_type: str) -> SurnameCorrector:
    """
    Load the trained model.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model
        
    Returns:
        Loaded SurnameCorrector instance
    """
    corrector = SurnameCorrector(model_type=model_type, model_path=model_path)
    
    # Check if reference surnames were loaded
    if not corrector.all_surnames:
        # Try to load from the same directory
        surnames_path = os.path.join(os.path.dirname(model_path), 'reference_surnames.pkl')
        if os.path.exists(surnames_path):
            with open(surnames_path, 'rb') as f:
                corrector.all_surnames = pickle.load(f)
    
    return corrector


def process_file(file_path: str, corrector: SurnameCorrector, 
                confidence_threshold: float) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single transcription file.
    
    Args:
        file_path: Path to the transcription file
        corrector: SurnameCorrector instance
        confidence_threshold: Minimum confidence for applying corrections
        
    Returns:
        Tuple of (corrected text, list of corrections)
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Correct the text
    corrected_text, corrections = corrector.correct_text(text, confidence_threshold)
    
    return corrected_text, corrections


def process_directory(directory: str, corrector: SurnameCorrector, 
                     confidence_threshold: float, output_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all text files in a directory.
    
    Args:
        directory: Path to directory with transcription files
        corrector: SurnameCorrector instance
        confidence_threshold: Minimum confidence for applying corrections
        output_dir: Directory to save corrected transcriptions
        
    Returns:
        Dictionary mapping filenames to lists of corrections
    """
    all_corrections = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # Process the file
            corrected_text, corrections = process_file(file_path, corrector, confidence_threshold)
            
            # Save corrected text
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
            
            # Store corrections
            all_corrections[filename] = corrections
    
    return all_corrections


def generate_correction_report(all_corrections: Dict[str, List[Dict[str, Any]]], 
                              output_path: str) -> None:
    """
    Generate a report of all corrections made.
    
    Args:
        all_corrections: Dictionary mapping filenames to lists of corrections
        output_path: Path to save the report
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Surname Correction Report\n\n")
        
        # Count total corrections
        total_files = len(all_corrections)
        total_corrections = sum(len(corrections) for corrections in all_corrections.values())
        
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Total corrections made: {total_corrections}\n\n")
        
        # Report corrections by file
        for filename, corrections in all_corrections.items():
            if not corrections:
                continue
                
            f.write(f"## {filename}\n\n")
            f.write("| Original | Corrected | Confidence |\n")
            f.write("|----------|-----------|------------|\n")
            
            for correction in corrections:
                f.write(f"| {correction['original']} | {correction['corrected']} | {correction['confidence']:.4f} |\n")
            
            f.write("\n")


def main():
    """Main function to correct surnames in transcribed text."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    corrector = load_model(args.model, args.model_type)
    
    # Check if we need to load reference data
    if args.reference_data and not corrector.all_surnames:
        print(f"Loading reference data from {args.reference_data}")
        reference_df = load_reference_data(args.reference_data)
        if not reference_df.empty:
            names, surnames = extract_names_surnames(reference_df)
            corrector.set_reference_surnames(surnames)
    
    # Check if we have reference surnames
    if not corrector.all_surnames:
        print("Error: No reference surnames available")
        return
    
    print(f"Loaded {len(corrector.all_surnames)} reference surnames")
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        print(f"Processing directory: {args.input}")
        all_corrections = process_directory(
            args.input, corrector, args.confidence, args.output
        )
        print(f"Processed {len(all_corrections)} files")
        
        # Count total corrections
        total_corrections = sum(len(corrections) for corrections in all_corrections.values())
        print(f"Made {total_corrections} corrections")
        
        # Generate report if requested
        if args.report:
            report_path = os.path.join(args.output, 'correction_report.md')
            print(f"Generating correction report: {report_path}")
            generate_correction_report(all_corrections, report_path)
    else:
        # Process single file
        print(f"Processing file: {args.input}")
        corrected_text, corrections = process_file(
            args.input, corrector, args.confidence
        )
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Save corrected text
        output_filename = os.path.basename(args.input)
        output_path = os.path.join(args.output, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_text)
        
        print(f"Made {len(corrections)} corrections")
        print(f"Saved corrected text to {output_path}")
        
        # Generate report if requested
        if args.report:
            report_path = os.path.join(args.output, 'correction_report.md')
            print(f"Generating correction report: {report_path}")
            generate_correction_report({output_filename: corrections}, report_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
